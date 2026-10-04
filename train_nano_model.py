"""T2 训练入口：tiny-shakespeare char-LM，nano-deepspeed 引擎，dp（ZeRO 0/1/2）与 tp 双形态。

用法：
    单进程        python train_nano_model.py --data data/input.txt --out runs/v2 --steps 200
    2-proc DP     torchrun --nproc-per-node 2 train_nano_model.py --zero-stage {0,1,2} ...
    2-proc TP     torchrun --nproc-per-node 2 train_nano_model.py --tp-size 2 ...

约定：
- positions 由训练侧显式给 arange(seq_len)——训练侧无序列状态簿记；部署侧
  （vllm）由 engine 拥有状态并产 positions，两侧接口同构（设计文档 §3）；
- TP 模式：config.gradient_clipping=0，边界 step 前用 clip_grad_norm_tp_ 外置裁剪
  （内嵌裁剪在 TP 下只算局部范数，且 replicated 梯度不可跨 rank 求和——双计）；
  TP 占满 world，优化器 DP 组为单例组（防 stage0 对 sharded 梯度跨 rank 求和破坏切分）；
- checkpoint 布局与 resume 语义见 nano_deepspeed/checkpoint.py 模块 docstring。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nano_model.config import NanoConfig
from nano_model.model import NanoModel
from nano_model.tokenizer import CharTokenizer

from nano_deepspeed import initialize
from nano_deepspeed.checkpoint import load_checkpoint, save_checkpoint
from nano_deepspeed.distributed import init_distributed
from nano_deepspeed.tensor_parallel import clip_grad_norm_tp_, init_tp_groups, reduce_tp_grads_
from nano_deepspeed.tp_nano_model import TPNanoModel, export_full_state_dict, load_tp_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/input.txt"))
    p.add_argument("--out", type=Path, required=True, help="运行产物目录（loss 曲线 / run_meta）")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8, help="每 rank micro batch")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=200, help="optimizer step 数（非 micro step）")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--zero-stage", type=int, default=0, choices=(0, 1, 2))
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--ckpt-dir", type=Path, default=None, help="默认 <out>/ckpt")
    p.add_argument("--save-every", type=int, default=0, help="按 opt step 间隔存 ckpt；0 = 只存最终")
    p.add_argument("--resume", type=Path, default=None, help="ckpt tag 目录（含 meta.json）")
    p.add_argument("--init-ckpt", type=Path, default=None, help="C1 目录作为初始权重（V3/V7/V8 可比性）")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="cuda 不可用时 fail-fast 退出，不静默回退（静默回退 = K8s 验证假绿）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true", help="训练集顺序遍历（单进程 vs TP 对比实验用）")
    p.add_argument("--probe-partition", action="store_true", help="V4：打印 ZeRO 分区账后退出")
    p.add_argument("--probe-forward", type=Path, default=None, help="V7：对照 ref logits/state 后退出（需 --init-ckpt）")
    p.add_argument("--probe-grad", type=Path, default=None, help="V8a：对照单进程参考梯度后退出（需 --init-ckpt）")
    return p.parse_args()


class BlockDataset(Dataset):
    """1D token 流 → 定长 block；X=block[:, :-1]，Y=block[:, 1:]（严格 shift 对齐）。"""

    def __init__(self, ids: torch.Tensor, seq_len: int) -> None:
        self.block = seq_len + 1
        self.n = (ids.numel() - 1) // self.block
        assert self.n > 0, f"语料太短：{ids.numel()} tokens < seq_len+1={self.block}"
        self.ids = ids[: self.n * self.block]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[i * self.block: (i + 1) * self.block]  # [L+1]
        return chunk[:-1], chunk[1:]  # ([L], [L]) Y[t] = X[t+1]


def positions_for(batch_size: int, seq_len: int) -> torch.Tensor:
    """训练侧 positions：arange（无序列状态簿记；部署侧由 engine 拥有状态后显式产出）。"""
    return torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)


def micro_loss(module, x: torch.Tensor, y: torch.Tensor, seq_len: int, vocab: int) -> torch.Tensor:
    positions = positions_for(x.size(0), seq_len).to(x.device)
    hidden = module(x, positions)
    logits = module.compute_logits(hidden)
    return F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(), y[:, 1:].reshape(-1))


def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _repo_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate(module, loader: DataLoader, seq_len: int, vocab: int) -> float:
    """token 级平均 loss。DP 模式仅 rank0 调用（无集合通信）；TP 模式全 rank 调用（前向含 AR）。"""
    module.eval()
    device = next(module.parameters()).device
    total, ntok = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        positions = positions_for(x.size(0), seq_len).to(device)
        hidden = module(x, positions)
        logits = module.compute_logits(hidden)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(), y[:, 1:].reshape(-1), reduction="sum")
        total += loss.item()
        ntok += y[:, 1:].numel()
    module.train()
    return total / ntok


def probe_partition(engine, stage: int, tp_size: int) -> None:
    """V4：ZeRO 分区账结构断言（非 RSS 账）。

    - stage 0：master/动量/二阶矩全量副本（4Ψ 中的 optim 状态 3 份全量）；
    - stage 1/2：partition_size × dp_world == aligned_total（对齐填充含在 aligned 内）；
      stage 1 梯度全量缓冲、stage 2 梯度分区缓冲；
    - TP 模式优化器 DP world == 1（单例组），分区退化为全量——这是设计决定（防
      stage0 对 sharded 梯度跨 rank 求和），此处断言生效。
    """
    optim = engine.optimizer
    dp_world = optim.world_size
    expected_dp = 1 if tp_size > 1 else (dist.get_world_size() if dist.is_initialized() else 1)
    assert dp_world == expected_dp, f"optimizer DP world {dp_world} != expected {expected_dp}"
    total_params = sum(fg.total_numel for fg in optim._flats)
    model_params = sum(p.numel() for p in engine.module.parameters())
    assert total_params == model_params, f"flat 覆盖 {total_params} != 模型参数 {model_params}"
    for fg in optim._flats:
        if stage == 0:
            assert fg.fp32_master_full.numel() == fg.aligned_total
            assert fg.exp_avg_full.numel() == fg.aligned_total
            assert fg.exp_avg_sq_full.numel() == fg.aligned_total
        else:
            assert fg.partition_size * dp_world == fg.aligned_total, "分区账不平"
            assert fg.fp32_master_shard.numel() == fg.partition_size
            if stage == 1:
                assert fg.grad_full_fp32.numel() == fg.aligned_total
            else:
                assert fg.grad_partition_fp32.numel() == fg.partition_size
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"V4 PASS rank={rank} stage={stage} dp_world={dp_world} tp={tp_size} "
          f"params={total_params} partition={optim._flats[0].partition_size} aligned={optim._flats[0].aligned_total}")


@torch.no_grad()
def probe_forward(engine, ref_path: Path, init_ckpt: Path) -> None:
    """V7：TP 前向等价 + 导出还原检查（对 C1 全量权重）。"""
    model = engine.module
    ref = torch.load(ref_path, weights_only=False)
    ids = ref["ids"]
    hidden = model(ids, positions_for(ids.size(0), ids.size(1)))
    logits = model.compute_logits(hidden)
    logit_diff = (logits - ref["logits"]).abs().max().item()

    state = export_full_state_dict(model)
    c1 = load_file(str(Path(init_ckpt) / "model.safetensors"))
    weight_diff = max((state[k].float() - c1[k].float()).abs().max().item() for k in c1)

    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"V7 rank={rank} max_logit_diff={logit_diff:.3e} max_weight_diff={weight_diff:.3e}")
    assert logit_diff < 1e-5, f"TP 前向与单进程参考偏差超限：{logit_diff:.3e}"
    assert weight_diff < 1e-5, f"TP 导出与 C1 权重偏差超限：{weight_diff:.3e}"
    if rank == 0:
        print("V7 PASS")


def probe_grad(engine, ref_path: Path, tp_group) -> None:
    """V8a：TP 梯度级等价（TP 训练等价的主判据——确定性、无混沌放大）。

    同一 batch、同一 loss 公式下，tp=2 各参数梯度 vs 单进程参考：
    sharded 参数 all_gather 还原全量后比较，replicated 参数（各 rank 恒等）直接比较。
    per-tensor rel = max|Δ| / max|g_ref|，判据 < 1e-3：ulp 级数值路径差预期 ~1e-5；
    真实 TP 数学 bug（如 f/g 算子漏 sum、切分维度错）rel≈O(1) 第一步即爆。
    200 步曲线比对（V8b）只保留终点收敛语义，混沌敏感性不再承担语义判定。
    """
    model = engine.module
    ref = torch.load(ref_path, weights_only=False)
    x = ref["ids"]
    hidden = model(x, positions_for(x.size(0), x.size(1)))
    logits = model.compute_logits(hidden)[:, :-1]
    loss = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())
    loss.backward()
    reduce_tp_grads_(engine.module, tp_group)  # qk-norm 梯度局部部分和 → 与训练路径同款规约

    shard_dim: dict[str, int] = {}  # 参数名 → 分片维度（仅 sharded 参数有）
    for mname, m in model.named_modules():
        sd = getattr(m, "tp_shard_dim", None)
        if sd is not None:
            for pname, _ in m.named_parameters(recurse=False):
                shard_dim[f"{mname}.{pname}"] = sd

    worst = 0.0
    worst_by: list[tuple[float, str]] = []
    for name, p in model.named_parameters():
        g = p.grad
        if name in shard_dim:
            parts = [torch.empty_like(g) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, g.contiguous())
            g = torch.cat(parts, dim=shard_dim[name])
        rel = ((g - ref["grads"][name]).abs().max()
               / ref["grads"][name].abs().max().clamp_min(1e-12)).item()
        worst_by.append((rel, name))
        worst = max(worst, rel)

    rank = dist.get_rank()
    worst_by.sort(reverse=True)
    for rel, name in worst_by[:5]:
        print(f"V8a rank={rank} grad_rel={rel:.3e} {name}")
    print(f"V8a rank={rank} max_grad_rel={worst:.3e}")
    assert worst < 1e-3, f"TP 梯度与单进程参考相对偏差超限：{worst:.3e}"
    if rank == 0:
        print("V8a PASS")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device cuda but torch.cuda.is_available() is False ({torch.version.cuda=})")
    # 显式 gloo 且统一在此预初始化：GPU pod（cuda 可用）上 engine 懒初始化默认选 nccl，
    # 而 loss 日志 all_reduce 是 CPU tensor（nccl 不支持 → V9 实测 RuntimeError）。
    # NCCL 留给 T5 由消费者显式选择。tp 建模需要 tp_group，也必须先于 initialize 懒初始化。
    if not dist.is_initialized():
        init_distributed("gloo")
    tp_group = init_tp_groups(args.tp_size)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = max(1, int(tokens.numel() * args.val_ratio))
    train_ids, val_ids = tokens[:-n_val], tokens[-n_val:]  # nanoGPT 约定：val 取尾部

    torch.manual_seed(args.seed)
    if args.init_ckpt:
        model = load_tp_model(args.init_ckpt, tp_group) if args.tp_size > 1 else NanoModel.load_pretrained(args.init_ckpt)
        config = model.config
    else:
        config = NanoConfig(vocab_size=tokenizer.vocab_size)
        model = TPNanoModel(config, tp_group) if args.tp_size > 1 else NanoModel(config)
    model.to(device)  # 引擎 flat buffer / 优化器状态跟随参数 device（zero_optimizer.py device 无关性）

    ds_cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "zero_optimization": {"stage": args.zero_stage},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": args.weight_decay},
        },
        # TP 模式内嵌裁剪关闭（局部范数无意义），由循环在边界 step 前外置裁剪
        "gradient_clipping": 0.0 if args.tp_size > 1 else args.clip,
    }
    kwargs = {}
    if dist.is_initialized() and args.tp_size > 1:
        # TP 占满 world，优化器 DP 组 = 单例组：world_size=1 使 stage0/1/2 的跨 rank
        # 梯度 all-reduce 全部退化为无操作（sharded 梯度跨 rank 求和 = 破坏切分语义）
        kwargs["data_parallel_group"] = dist.new_group([rank])
    engine, optim, _, _ = initialize(model=model, config=ds_cfg, **kwargs)

    # world 信息在 initialize 懒初始化 dist 之后才可靠（tp=1 时上面取值是 stale 的 1）
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if args.probe_partition:
        probe_partition(engine, args.zero_stage, args.tp_size)
        return
    if args.probe_forward is not None:
        probe_forward(engine, args.probe_forward, args.init_ckpt)

    if args.probe_grad is not None:
        probe_grad(engine, args.probe_grad, tp_group)
        return

    # ---- 数据管线 ----
    train_ds = BlockDataset(train_ids, args.seq_len)
    val_ds = BlockDataset(val_ids, args.seq_len)
    sampler = None
    if dist.is_initialized() and args.tp_size == 1:
        sampler = DistributedSampler(train_ds)  # DP 切数据；TP 模式各 rank 同数据无 sampler
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None and not args.no_shuffle), drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    if is_rank0():
        print(f"data: vocab={tokenizer.vocab_size} train_blocks={len(train_ds)} val_blocks={len(val_ds)} "
              f"world={world_size} tp={args.tp_size} zero_stage={args.zero_stage}")

    # ---- resume / 日志 ----
    out_dir = args.out
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else out_dir / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, resume_micro, opt_step = 0, 0, 0
    if args.resume is not None:
        meta = load_checkpoint(args.resume, engine)
        assert meta["micro_in_epoch"] % args.grad_accum == 0, "resume 点必须在 opt step 边界"
        start_epoch, resume_micro, opt_step = meta["epoch"], meta["micro_in_epoch"], meta["step"]
        if is_rank0():
            print(f"resumed from {args.resume}: step={opt_step} epoch={start_epoch} micro={resume_micro}")
    elif is_rank0():
        (out_dir / "run_meta.json").write_text(json.dumps({
            "args": vars(args) | {"data": str(args.data), "out": str(out_dir)},
            "world_size": world_size, "tp_size": args.tp_size, "zero_stage": args.zero_stage,
            "git_sha": _repo_sha(),
        }, indent=2, ensure_ascii=False, default=str))

    log_path = out_dir / "loss_train.jsonl"
    log_f = open(log_path, "a" if args.resume is not None else "w", encoding="utf-8")

    def save_tag(tag_step: int, epoch: int, micro_in_epoch: int) -> Path:
        tag = ckpt_dir / f"step{tag_step:06d}"
        save_checkpoint(tag, engine, step=tag_step, epoch=epoch, micro_in_epoch=micro_in_epoch, tokenizer=tokenizer)
        return tag

    # ---- 训练循环 ----
    # ---- epoch 对齐 grad_accum（resume 决定性前提）----
    # 每 epoch 只消费 L_aligned 个 micro（尾部非整组 micro 丢弃，直通/续训丢弃同一批）：
    # 否则 accumulation 组跨 epoch 边界，opt step 边界落在 micro_in_epoch 非 4 倍数位置，
    # resume 断言（本地 _micro_steps 从 0 起数与全局组对齐）必挂——200 steps 实测复现。
    epoch_batches = len(train_loader)  # drop_last=True → 每 rank 整 batch 数
    aligned_batches = epoch_batches - epoch_batches % args.grad_accum
    assert aligned_batches > 0, f"epoch micro 数过小：{epoch_batches} batches < grad_accum={args.grad_accum}"

    finished = False
    for epoch in range(start_epoch, 10 ** 9):
        if sampler is not None:
            sampler.set_epoch(epoch)
        micro_in_epoch = 0
        for i, (x, y) in enumerate(train_loader):
            if i >= aligned_batches:
                break  # epoch 对齐截断：尾部非整组 micro 不消费（含快进段在内，全 epoch 一致）
            if epoch == start_epoch and micro_in_epoch < resume_micro:
                micro_in_epoch += 1  # 决定性快进：同 epoch 内跳过已消费的 micro batch
                continue
            x, y = x.to(device), y.to(device)
            loss = micro_loss(engine.module, x, y, args.seq_len, config.vocab_size)
            engine.backward(loss)
            micro_in_epoch += 1

            # engine._micro_steps 是累计 micro 数（step() 常规边界不清零），边界判断取模，
            # 与 engine.step() 内部 at_boundary 逻辑一致
            boundary = engine._micro_steps % args.grad_accum == 0
            if boundary and args.tp_size > 1:
                reduce_tp_grads_(engine.module, tp_group)  # qk-norm 梯度是局部部分和，先规约
                if args.clip > 0:
                    clip_grad_norm_tp_(engine.module, engine.optimizer, args.clip, tp_group)
            engine.step()  # 非 boundary 自动 no-op（边界感知见 engine.py step()）
            if not boundary:
                continue

            opt_step += 1
            # CPU tensor 后再做跨 rank AR：gloo 对 cpu tensor 无歧义（cuda tensor 亦可但隐式拷贝）
            loss_log = loss.detach().float().cpu()
            if args.tp_size == 1 and dist.is_initialized() and dist.get_world_size() > 1:
                # DP 各 rank 的 micro loss 是本地 batch 的值（梯度才跨 rank 平均），
                # 日志取跨 rank 均值，使曲线与单进程可比
                dist.all_reduce(loss_log)
                loss_log /= dist.get_world_size()
            if is_rank0():
                log_f.write(json.dumps({"step": opt_step, "epoch": epoch, "loss": loss_log.item()}) + "\n")
                log_f.flush()
                if opt_step % 20 == 0:
                    print(f"step {opt_step}/{args.steps} loss {loss_log.item():.4f}")
            if args.save_every > 0 and opt_step % args.save_every == 0:
                save_tag(opt_step, epoch, micro_in_epoch)
            if opt_step >= args.steps:
                finished = True
                break
        if finished:
            break

    final_tag = save_tag(opt_step, epoch, micro_in_epoch)
    val_loss = None
    if args.tp_size > 1 or is_rank0():  # TP：全 rank 参与（前向含 AR）；DP/单进程：仅 rank0
        val_loss = evaluate(engine.module, val_loader, args.seq_len, config.vocab_size)
    if is_rank0():
        (out_dir / "loss_val.json").write_text(json.dumps(
            {"step": opt_step, "loss": val_loss, "ppl": None if val_loss is None else math.exp(val_loss),
             "ckpt": str(final_tag)}, indent=2))
        print(f"done: steps={opt_step} val_loss={val_loss:.4f} ckpt={final_tag}")
    log_f.close()


if __name__ == "__main__":
    main()
