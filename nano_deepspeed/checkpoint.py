"""训练 checkpoint——C1 模型导出 + 每 rank 优化器/RNG 状态 + meta 校验。

目录布局（<ckpt_dir>/ 由调用方给到 tag 级，如 runs/x/ckpt/step000100）：

    model/                 # C1 导出：model.safetensors(bf16) + config.json —— train→serve 交接物
    vocab.json             # tokenizer itos（可选，save 时传入才写）
    optim_rank{r}.pt       # ZeroOptimizer.state_dict()：fp32 master + 动量 + 二阶矩
    rng_rank{r}.pt         # torch RNG + python random 状态（决定性 resume）
    meta.json              # step / epoch / micro_in_epoch / world_size / zero_stage / tp_size / git_sha

职责边界（doc/topics/t2_nano_deepspeed.md §4）：
- resume 不读 model/——bf16 有损；fp32 master 权重在 optimizer state_dict 里，
  load_state_dict 会回填 flat_param → 参数视图（zero_optimizer.py load 尾部），
  V5 resume 等价性（tol 1e-6）依赖这条路径；
- 每 rank 落自己的 optim/rng 文件（文件名用全局 rank：TP 模式下 DP 是单例组，
  dp rank 恒为 0，不能做文件名）；load 侧对 stage/world_size/tp_size 逐一校验
  （resume 约定：同 world_size 同 stage 同 tp 布局）。
"""

from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from nano_model.model import DTYPE_MAP

from .tp_nano_model import TPNanoModel, export_full_state_dict, tp_world_size


def _repo_sha() -> str | None:
    """nano-deepspeed fork 仓 HEAD（best effort，用于 meta 追溯）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _world_rank() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


@torch.no_grad()
def save_checkpoint(
    ckpt_dir: str | Path,
    engine,
    *,
    step: int,
    epoch: int,
    micro_in_epoch: int = 0,
    tokenizer=None,
) -> Path:
    """保存到 <ckpt_dir>（含 model/ 的 C1 导出与每 rank 状态）。

    TP 模式：export_full_state_dict 是集合通信，全 rank 都要参与，仅 rank0 落盘；
    DP 模式：仅 rank0 导出模型，各 rank 各自落 optim/rng。结尾 barrier 保证
    save 返回时全部文件可见（Job 收尾即存即退的场景）。
    """
    ckpt_dir = Path(ckpt_dir)
    model = engine.module
    is_tp = isinstance(model, TPNanoModel)
    world_size, rank = _world_rank()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    full_state = None
    if is_tp:
        full_state = export_full_state_dict(model)  # 集合操作：所有 TP rank 必须到达这里

    if rank == 0:
        model_dir = ckpt_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        if is_tp:
            dtype = DTYPE_MAP[model.config.dtype]
            state = {
                k: v.detach().to(dtype).cpu()
                for k, v in full_state.items()
                if not k.startswith("lm_head.")  # tied 权重按 C1 不落盘（named_parameters 已去重，防御性过滤）
            }
            save_file(state, str(model_dir / "model.safetensors"), metadata={"format": "pt"})
            model.config.save(model_dir / "config.json")
        else:
            model.save_pretrained(model_dir)
        if tokenizer is not None:
            tokenizer.save(ckpt_dir / "vocab.json")

        meta = {
            "format_version": 1,
            "step": int(step),
            "epoch": int(epoch),
            "micro_in_epoch": int(micro_in_epoch),
            "world_size": int(world_size),
            "zero_stage": int(engine.optimizer.stage),
            "tp_size": int(tp_world_size(model.tp_group)) if is_tp else 1,
            "git_sha": _repo_sha(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    torch.save(engine.optimizer.state_dict(), ckpt_dir / f"optim_rank{rank}.pt")
    torch.save(
        {"torch": torch.get_rng_state(), "python": random.getstate()},
        ckpt_dir / f"rng_rank{rank}.pt",
    )
    if dist.is_initialized():
        dist.barrier()
    return ckpt_dir


def load_checkpoint(ckpt_dir: str | Path, engine) -> dict:
    """恢复 optim（含 fp32 master 回填参数）与 RNG，返回 meta。

    校验 resume 前提：同 zero_stage / 同 world_size / 同 tp_size。
    不加载 model/——参数权重由 optimizer.load_state_dict 的 master 回填路径恢复。
    """
    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    model = engine.module
    is_tp = isinstance(model, TPNanoModel)
    world_size, rank = _world_rank()

    if int(meta["zero_stage"]) != int(engine.optimizer.stage):
        raise ValueError(f"zero_stage mismatch: ckpt={meta['zero_stage']} current={engine.optimizer.stage}")
    if int(meta["world_size"]) != int(world_size):
        raise ValueError(f"world_size mismatch: ckpt={meta['world_size']} current={world_size}")
    tp_size = int(tp_world_size(model.tp_group)) if is_tp else 1
    if int(meta["tp_size"]) != int(tp_size):
        raise ValueError(f"tp_size mismatch: ckpt={meta['tp_size']} current={tp_size}")

    optim_state = torch.load(ckpt_dir / f"optim_rank{rank}.pt", map_location="cpu", weights_only=False)
    engine.optimizer.load_state_dict(optim_state)  # 内部带 stage/world/rank/shape 校验 + master→flat_param 回填

    rng = torch.load(ckpt_dir / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    return meta
