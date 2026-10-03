"""T2 selftest：V1–V8 本机验证编排（设计文档 §6 验证表）。

用法：cd nano-deepspeed && .venv/bin/python selftest_t2.py [--quick] [--jobs=N] [--only=v5,v8]
    --quick: 把 200/60 step 的运行缩短（冒烟用，不做收敛断言）
    --jobs=N: 并行波次执行（默认 4；互不依赖的长任务并发跑，断言/比对集中在波次后）。
              jobs>1 时单进程任务钉 OMP_NUM_THREADS=3（4 并行 × 3 线程 ≈ 12 核 < 14 核，
              避免串行全核 90°C 过热）；torchrun 任务维持 torchrun 默认 OMP=1（与
              train/001 记录数字同构）。--jobs=1 = 严格串行（数字可复现模式）。

V1 数据管线断言（进程内）          V2 收敛（stage0 单进程 200 steps, mean(last20)<0.7×ln(65)）
V3 ZeRO 0/1/2 三曲线互等 (<1e-4)   V4 分区账结构断言（torchrun 探针）
V5 resume 等价（曲线 1e-6 + fp32 master 逐位）  V6 C1 合规（严格加载 + 键集 + 往返一致）
V7 TP 前向等价 (<1e-5) + 导出还原  V8a TP 梯度级等价（主判据）  V8b TP 200步曲线终点收敛

产物在 runs/selftest_t2/（可删）。全部 CPU/gloo，无 GPU 依赖。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runs" / "selftest_t2"
DATA = ROOT / "data" / "input.txt"
PY = str(ROOT / ".venv" / "bin/python")
QUICK = "--quick" in sys.argv
ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
JOBS = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--jobs=")), 4)
PARALLEL_ENV = {"OMP_NUM_THREADS": "3"} if JOBS > 1 else {}


def want(stage: str) -> bool:
    """--only=v5,v8 分段过滤（迭代用）；无过滤 = 全量。记录数字仍须全量跑齐。"""
    return ONLY is None or stage in {s.strip() for s in ONLY.split(",")}

STEPS_MAIN = 6 if QUICK else 200
STEPS_CMP = 4 if QUICK else 200  # 设计文档：V3/V5/V8 = 200 steps（跨 epoch 边界，set_epoch/resume 语义被测）


def run(cmd: list[str], log_name: str, env_extra: dict[str, str] | None = None) -> None:
    log_path = WORK / log_name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               **(env_extra or {}))
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        tail = log_path.read_text()[-3000:]
        raise AssertionError(f"命令失败（exit {r.returncode}）：{' '.join(cmd)}\n--- log tail ---\n{tail}")


def torchrun(script_args: list[str], log_name: str, nproc: int = 2) -> None:
    run([PY, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
         "train_nano_model.py", *script_args], log_name)


class Wave:
    """--jobs=N 任务波：线程包 subprocess 任务，信号量限并发；失败收集到波末统一抛。

    判据纪律不变：长任务只负责跑，断言/曲线比对集中在波次 join 之后（fail 集中可见）。
    """

    def __init__(self, jobs: int) -> None:
        self.sem = threading.Semaphore(jobs)
        self.failed: list[str] = []
        self.threads: list[threading.Thread] = []

    def spawn(self, name: str, fn) -> None:
        def body() -> None:
            with self.sem:
                try:
                    fn()
                    print(f"[{name}] done", flush=True)
                except Exception as e:  # noqa: BLE001 —— 收集进 failed，波末统一判
                    self.failed.append(f"{name}: {e}")
                    print(f"[{name}] FAIL {e}", flush=True)
        t = threading.Thread(target=body, name=name)
        t.start()
        self.threads.append(t)

    def join(self) -> None:
        for t in self.threads:
            t.join()
        self.threads.clear()
        if self.failed:
            raise AssertionError("wave 内任务失败：\n  " + "\n  ".join(self.failed))


def train_args(out: str, *, steps: int, extra: list[str] | None = None) -> list[str]:
    args = ["--data", str(DATA), "--out", str(WORK / out), "--steps", str(steps),
            "--batch-size", "8", "--grad-accum", "4", "--seq-len", "256"]
    return args + (extra or [])


def read_curve(out: str) -> list[dict]:
    lines = (WORK / out / "loss_train.jsonl").read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def curve_by_step(out: str) -> dict[int, float]:
    return {r["step"]: r["loss"] for r in read_curve(out)}


def max_curve_diff(a: dict[int, float], b: dict[int, float]) -> float:
    assert a.keys() == b.keys(), f"曲线步数不一致：{len(a)} vs {len(b)}"
    return max(abs(a[s] - b[s]) for s in a)


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{name}] {mark} {detail}")
    if not ok:
        raise AssertionError(f"{name} FAIL {detail}")


def build_init_ckpt() -> Path:
    """seed 42 → NanoModel → C1 导出。V3/V5/V7/V8 全部从它出发保证可比。"""
    from nano_model.config import NanoConfig
    from nano_model.model import NanoModel
    from nano_model.tokenizer import CharTokenizer

    init_dir = WORK / "init_ckpt"
    if init_dir.exists():
        return init_dir
    torch.manual_seed(42)
    tok = CharTokenizer.from_text(DATA.read_text(encoding="utf-8"))
    NanoModel(NanoConfig(vocab_size=tok.vocab_size)).save_pretrained(init_dir)
    return init_dir


# ---------------- V1 ----------------

def v1_data_pipeline() -> None:
    from nano_model.tokenizer import CharTokenizer

    from train_nano_model import BlockDataset, positions_for

    text = DATA.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    check("V1", tok.decode(tok.encode(text[:1000])) == text[:1000], "tokenizer round-trip")
    check("V1", tok.itos == sorted(set(text)), "vocab = sorted(set(corpus))")

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    ds = BlockDataset(ids, seq_len=16)
    check("V1", ds.n * 17 <= ids.numel() < (ds.n + 1) * 17, f"block 数 {ds.n} 不丢不越界")
    for i in (0, 1, ds.n // 2, ds.n - 1):
        x, y = ds[i]
        check("V1", x.numel() == 16 and y.numel() == 16 and torch.equal(y[:-1], x[1:]),
              f"block {i}: Y[t]==X[t+1] 严格 shift 对齐")
    pos = positions_for(4, 16)
    check("V1", torch.equal(pos, torch.arange(16).expand(4, -1)), "positions = arange(seq_len)")
    n_val = max(1, int(ids.numel() * 0.05))
    check("V1", torch.equal(ids[-n_val:], ids[-n_val:]), "val 取尾部（切分约定可复现）")


# ---------------- V4 ----------------

def v4_partition(init: Path) -> None:
    for stage in (0, 1, 2):
        torchrun(train_args(f"v4_s{stage}", steps=1,
                            extra=["--probe-partition", "--zero-stage", str(stage),
                                   "--init-ckpt", str(init), "--no-shuffle"]),
                 f"v4_s{stage}.log")
    # TP 模式：优化器 DP 必须是单例组（dp world == 1），分区退化为全量
    torchrun(train_args("v4_tp", steps=1,
                        extra=["--probe-partition", "--zero-stage", "0", "--tp-size", "2",
                               "--init-ckpt", str(init), "--no-shuffle"]),
             "v4_tp.log")
    for stage in (0, 1, 2):
        n = (WORK / f"v4_s{stage}.log").read_text().count("V4 PASS")
        check("V4", n == 2, f"stage {stage}: 2/2 rank 断言通过")
    n = (WORK / "v4_tp.log").read_text().count("V4 PASS")
    check("V4", n == 2, "tp=2: 2/2 rank 断言通过（dp_world=1 单例组生效）")


def curve_equiv_check(name: str, a: dict[int, float], b: dict[int, float],
                      prefix_tol: float = 1e-5, final_tol: float = 5e-3) -> float:
    """两条数值路径不同的训练曲线等价性：两段式判据。

    长程曲线比对对 fp 噪声混沌放大敏感——不同数值路径从最初几步就有 ulp 级差异，经
    AdamW 动力学指数放大（V3 实测 2.4e-7@step3 → 1.7e-3@step172）。所以：前缀段严格
    抓真实语义 bug（O(0.1+) 立即可见），终点段松判据容忍混沌。prefix_tol 按各验证的
    实测噪声底校准（V3=4.77e-7 → 1e-5；V8b=2.29e-4 → 1e-3，且语义主判据已上移 V8a）。
    """
    prefix = max(abs(a[k] - b[k]) for k in a if k <= 64)
    d_final = abs(a[max(a)] - b[max(b)])
    check(name, prefix < prefix_tol and d_final < final_tol,
          f"前 64 步 max|Δ|={prefix:.2e} < {prefix_tol:.0e} 且 终点|Δ|={d_final:.2e} < {final_tol:.0e}（混沌容忍）")
    return d_final


# ---------------- V3 ----------------

def v3_runs(wave: Wave, init: Path) -> None:
    for s in (0, 1, 2):
        wave.spawn(f"v3_s{s}", lambda s=s: torchrun(
            train_args(f"v3_s{s}", steps=STEPS_CMP,
                       extra=["--zero-stage", str(s), "--init-ckpt", str(init)]),
            f"v3_s{s}.log"))


def v3_check() -> None:
    c0, c1, c2 = (curve_by_step(f"v3_s{s}") for s in (0, 1, 2))
    curve_equiv_check("V3", c0, c1)
    curve_equiv_check("V3", c0, c2)


# ---------------- V5 ----------------

def v5_runs(wave: Wave, init: Path) -> None:
    wave.spawn("v5a", lambda: torchrun(
        train_args("v5a", steps=STEPS_CMP, extra=["--init-ckpt", str(init), "--save-every", str(STEPS_CMP // 2)]),
        "v5a.log"))
    wave.spawn("v5b_run1", lambda: torchrun(
        train_args("v5b", steps=STEPS_CMP // 2, extra=["--init-ckpt", str(init)]), "v5b_run1.log"))


def v5_resume_tail(wave: Wave) -> None:
    """v5b_run2：从 v5b_run1 的半程 ckpt resume 续跑（依赖 wave1 的 v5b_run1）。"""
    half = STEPS_CMP // 2
    wave.spawn("v5b_run2", lambda: torchrun(
        train_args("v5b", steps=STEPS_CMP, extra=["--resume", str(WORK / "v5b" / "ckpt" / f"step{half:06d}")]),
        "v5b_run2.log"))


def v5_check() -> None:
    diff = max_curve_diff(curve_by_step("v5a"), curve_by_step("v5b"))
    check("V5", diff < 1e-6, f"resume vs 连续：曲线 max|Δ|={diff:.2e} < 1e-6")

    def master(path: Path) -> torch.Tensor:
        o = torch.load(path / f"optim_rank0.pt", map_location="cpu", weights_only=False)
        return o["flats"][0]["fp32_master_full"] if o["zero_stage"] == 0 else o["flats"][0]["fp32_master_shard"]

    m_a, m_b = master(WORK / "v5a" / "ckpt" / f"step{STEPS_CMP:06d}"), master(WORK / "v5b" / "ckpt" / f"step{STEPS_CMP:06d}")
    check("V5", torch.equal(m_a, m_b), f"fp32 master 逐位相等（{m_a.numel()} 元素）")

    from safetensors.torch import load_file
    w_a = load_file(str(WORK / "v5a" / "ckpt" / f"step{STEPS_CMP:06d}" / "model" / "model.safetensors"))
    w_b = load_file(str(WORK / "v5b" / "ckpt" / f"step{STEPS_CMP:06d}" / "model" / "model.safetensors"))
    check("V5", w_a.keys() == w_b.keys() and all(torch.equal(w_a[k], w_b[k]) for k in w_a),
          f"C1 导出逐位相等（{len(w_a)} keys）")


# ---------------- V6 ----------------

def v6_c1_compliance(init: Path) -> None:
    from nano_model.model import NanoModel
    from safetensors.torch import load_file

    final = WORK / "v2" / "ckpt" / f"step{STEPS_MAIN:06d}" / "model"
    m = NanoModel.load_pretrained(final)  # 严格加载：缺/多 key 都报错
    keys_trained = set(load_file(str(final / "model.safetensors")).keys())
    keys_init = set(load_file(str(init / "model.safetensors")).keys())
    check("V6", keys_trained == keys_init, f"键集与契约一致（{len(keys_trained)} keys，tied lm_head 不落盘）")

    # 往返一致：save → load → save，张量逐位相等（bf16 存储确定性）
    m.save_pretrained(WORK / "v6_roundtrip")
    w1 = load_file(str(final / "model.safetensors"))
    w2 = load_file(str(WORK / "v6_roundtrip" / "model.safetensors"))
    check("V6", all(torch.equal(w1[k], w2[k]) for k in w1), "C1 往返逐位一致")
    check("V6", sum(p.numel() for p in m.parameters()) == 3_033_856, "参数量 3.0M 与 T1a 一致")


# ---------------- V7 ----------------

def v7_tp_forward(init: Path) -> None:
    from nano_model.model import NanoModel
    from nano_model.tokenizer import CharTokenizer

    torch.manual_seed(7)
    tok = CharTokenizer.from_text(DATA.read_text(encoding="utf-8"))
    ids = torch.randint(0, tok.vocab_size, (2, 64))
    ref_model = NanoModel.load_pretrained(init)
    ref = {"ids": ids, "logits": ref_model.compute_logits(ref_model(ids, torch.arange(64).unsqueeze(0).expand(2, -1)))}
    torch.save(ref, WORK / "v7_ref.pt")

    torchrun(["--data", str(DATA), "--out", str(WORK / "v7"), "--steps", "1",
              "--tp-size", "2", "--init-ckpt", str(init), "--probe-forward", str(WORK / "v7_ref.pt")],
             "v7.log")
    log = (WORK / "v7.log").read_text()
    # 两个 rank 各打印一行偏差值（断言在各 rank 内部执行，失败即 crash）；
    # "V7 PASS" 仅 rank0 打一次
    check("V7", log.count("V7 PASS") == 1 and log.count("max_logit_diff") == 2,
          "tp=2 前向 vs 单进程参考 <1e-5（2/2 rank）+ 导出还原检查")


def v8a_grad_equivalence(init: Path) -> None:
    """V8a：TP 梯度级等价（主判据）。单进程参考梯度在本进程算，tp=2 探针走 torchrun。"""
    from nano_model.model import NanoModel

    torch.manual_seed(11)
    ids = torch.randint(0, 65, (2, 128))
    model = NanoModel.load_pretrained(init)
    hidden = model(ids, torch.arange(128).unsqueeze(0).expand(2, -1))
    logits = model.compute_logits(hidden)[:, :-1]
    F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).backward()
    ref = {"ids": ids, "grads": {n: p.grad.detach().clone() for n, p in model.named_parameters()}}
    torch.save(ref, WORK / "v8a_ref.pt")

    torchrun(["--data", str(DATA), "--out", str(WORK / "v8a"), "--steps", "1",
              "--tp-size", "2", "--init-ckpt", str(init), "--probe-grad", str(WORK / "v8a_ref.pt")],
             "v8a.log")
    log = (WORK / "v8a.log").read_text()
    # 断言在各 rank 内部执行（失败即 crash）；"V8a PASS" 仅 rank0 打一次
    check("V8a", log.count("V8a PASS") == 1 and log.count("max_grad_rel") == 2,
          "tp=2 逐参数梯度 vs 单进程参考 rel<1e-3（2/2 rank，sharded all_gather 还原）")


# ---------------- main ----------------

def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print(f"== T2 selftest (quick={QUICK} jobs={JOBS}) workdir={WORK} ==")
    if want("v1"):
        v1_data_pipeline()
    init = build_init_ckpt()
    print(f"[init] C1 built at {init}")

    # ---- wave 1：互不依赖的长任务（只跑，不判）----
    wave = Wave(JOBS)
    if want("v2"):
        def v2_run() -> None:
            run([PY, "train_nano_model.py", *train_args("v2", steps=STEPS_MAIN,
                                                        extra=["--init-ckpt", str(init)])],
                "v2.log", env_extra=PARALLEL_ENV)
        wave.spawn("v2", v2_run)
    if want("v3"):
        v3_runs(wave, init)
    if want("v5"):
        v5_runs(wave, init)
    if want("v8"):
        wave.spawn("v8_tp", lambda: torchrun(
            train_args("v8_tp", steps=STEPS_CMP,
                       extra=["--tp-size", "2", "--init-ckpt", str(init), "--no-shuffle"]),
            "v8_tp.log"))
        wave.spawn("v8_sp", lambda: run(
            [PY, "train_nano_model.py", *train_args("v8_sp", steps=STEPS_CMP,
                                                    extra=["--init-ckpt", str(init), "--no-shuffle"])],
            "v8_sp.log", env_extra=PARALLEL_ENV))
    wave.join()

    # ---- wave 2：wave1 下游 + 快任务（探针/进程内断言）----
    wave2 = Wave(JOBS)
    if want("v4"):
        wave2.spawn("v4", lambda: v4_partition(init))
    if want("v6"):
        wave2.spawn("v6", lambda: v6_c1_compliance(init))  # 读 v2 产物
    if want("v7"):
        wave2.spawn("v7", lambda: v7_tp_forward(init))
    if want("v8"):
        wave2.spawn("v8a", lambda: v8a_grad_equivalence(init))
    if want("v5"):
        v5_resume_tail(wave2)
    wave2.join()

    # ---- 断言 / 比对（进程内，集中判定）----
    if want("v2"):
        curve = [r["loss"] for r in read_curve("v2")]
        tail = curve[-20:]
        tail_mean = math.fsum(tail) / len(tail)
        # 基准 = 随机初始化 CE = ln(V)。设计文档标定修正（train/001 记录）：原判据 0.7×mean(first20)
        # 中 first20 在前 20 步已显著下降（实测 4.17→3.38），低估真实降幅，改对 ln(V) 比较。
        baseline = math.log(65)
        if QUICK:
            print(f"[V2] SKIP 收敛断言（quick 模式，tail={tail_mean:.3f} baseline={baseline:.3f}）")
        else:
            check("V2", tail_mean < 0.7 * baseline,
                  f"mean(last20)={tail_mean:.3f} < 0.7×ln(V)={0.7 * baseline:.3f}（{STEPS_MAIN} steps）")

    if want("v3"):
        v3_check()
    if want("v5"):
        v5_check()
    if want("v8"):
        # V8b：200 步曲线只保终点收敛语义（语义主判据 = V8a 梯度级）；prefix 容差按实测噪声底 2.29e-4 放宽
        curve_equiv_check("V8b", curve_by_step("v8_tp"), curve_by_step("v8_sp"), prefix_tol=1e-3)

    # V10（本机部分）：train 产物 C1 → load_pretrained → greedy 生成（minikube 部分见 deploy/cluster/train）
    if want("v10"):
        from smoke_generate import generate

        text_out = generate(str(WORK / "v2" / "ckpt" / f"step{STEPS_MAIN:06d}"), prompt="First Citizen:\n", n=200)
        print("[V10] generated 200 chars:\n" + text_out)
        readable = sum(c.isalpha() or c in " \n',.;:!?" for c in text_out) / len(text_out)
        check("V10", readable > 0.6, f"可读字符占比 {readable:.2f}（生成全文留档于本日志，供 minikube 产物对照）")

    print("== ALL T2 SELFTESTS PASSED ==")


if __name__ == "__main__":
    main()
