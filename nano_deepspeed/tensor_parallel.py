"""Tensor Parallel（Megatron 列/行切）——T2 新增，基座完全没有 TP（api.py 显式 del mpu）。

语义锚点（生产对照，只读参照）：
- repos/vllm/vllm/model_executor/layers/linear.py:430  ColumnParallelLinear
- repos/vllm/vllm/model_executor/layers/linear.py:1623 RowParallelLinear（reduce_results → all-reduce :1775）

Megatron 通信规则（本实现的正确性核心）：
- Column-parallel（权重沿输出维切）：前向无通信；反向对输入梯度做 all-reduce（f 算子）。
  原因：输入在各 rank 复制，每 rank 只算出 dL/dx 的部分和，须求和后共享参数（如 RMSNorm）
  的梯度才一致。
- Row-parallel（权重沿输入维切）：前向 all-reduce 输出（g 算子）；反向恒等（各 rank 的
  输入本就是自己的分片，梯度天然局部）。
- f/g 成对出现，每 DecoderBlock 前向 2 次 all-reduce（o_proj 后、down_proj 后），
  反向 2 次（进入 q/k/v、进入 gate/up 之前的输入梯度）。

nano 约束：tp_group = 整个 world（TP 与 DP 两轴不联合）；embedding/lm_head replicated
（char vocab 太小，切分无学习价值）。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def init_tp_groups(tp_size: int):
    """建立 TP 通信组。nano 语义：tp_size 必须等于 world_size（TP 占满 world，DP=1）。

    返回值：tp_size == 1 时返回 None（单进程，通信全部退化为恒等，代码路径统一）。
    """
    if not dist.is_initialized():
        if tp_size == 1:
            return None
        raise RuntimeError("init_tp_groups(tp_size>1) requires torchrun-initialized dist")
    world_size = dist.get_world_size()
    if tp_size > 1 and tp_size != world_size:
        raise NotImplementedError(
            f"nano TP 仅支持 tp_size == world_size（TP×DP 联合不在范围）：tp={tp_size}, world={world_size}"
        )
    return dist.group.WORLD if tp_size > 1 else None


def tp_world_size(group) -> int:
    return dist.get_world_size(group) if group is not None else 1


def tp_rank(group) -> int:
    return dist.get_rank(group) if group is not None else 0


class _AllReduceBackward(torch.autograd.Function):
    """Megatron f 算子：前向恒等，反向 all-reduce 输入梯度（sum）。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if ctx.group is not None and dist.get_world_size(ctx.group) > 1:
            grad = grad.contiguous()
            dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _AllReduceForward(torch.autograd.Function):
    """Megatron g 算子：前向 all-reduce（sum），反向恒等。

    out-of-place（clone 后 reduce）：autograd 需要本 rank 的局部值算权重梯度，
    不能 in-place 破坏 version counter。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        if group is not None and dist.get_world_size(group) > 1:
            out = x.clone()
            dist.all_reduce(out, group=group)
            return out
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class ColumnParallelLinear(nn.Module):
    """Y = XW^T，W [out_features, in_features] 沿 out 维切：每 rank W[r] = W[r*s:(r+1)*s, :]。

    前向：局部 GEMM，输出为本 rank 分片 [*, out/tp]，无通信。
    反向：输入梯度经 f 算子 all-reduce（见模块 docstring）。
    用于：q_proj / k_proj / v_proj / gate_proj / up_proj。
    """

    def __init__(self, in_features: int, out_features: int, tp_group, bias: bool = False) -> None:
        super().__init__()
        tp = tp_world_size(tp_group)
        assert out_features % tp == 0, f"out_features {out_features} 不能整除 tp={tp}"
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_rank = out_features // tp
        self.tp_group = tp_group
        self.tp_shard_dim = 0  # 权重分片维度标记（checkpoint 导出/裁剪用）
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            raise NotImplementedError("nano 层全部 bias=False")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _AllReduceBackward.apply(x, self.tp_group)
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """Y = XW^T，W [out_features, in_features] 沿 in 维切：每 rank W[r] = W[:, r*s:(r+1)*s]。

    输入须已并行（input_is_parallel=True，来自上游 Column 分片）；
    前向：局部 GEMM 后 g 算子 all-reduce 求和；反向恒等。
    用于：o_proj / down_proj。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group,
        input_is_parallel: bool = True,
        reduce_output: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        tp = tp_world_size(tp_group)
        assert in_features % tp == 0, f"in_features {in_features} 不能整除 tp={tp}"
        assert input_is_parallel, "nano 只用并行输入形态（replicated 输入请用 Column+gather，未实现）"
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_rank = in_features // tp
        self.tp_group = tp_group
        self.reduce_output = reduce_output
        self.tp_shard_dim = 1  # 权重分片维度标记
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            raise NotImplementedError("nano 层全部 bias=False")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if self.reduce_output:
            out = _AllReduceForward.apply(out, self.tp_group)
        return out


def _tp_sharded_param_ids(model: nn.Module) -> set[int]:
    """收集挂在 Column/RowParallelLinear（宿主带 tp_shard_dim 标记）下的参数 id。"""
    ids: set[int] = set()
    for module in model.modules():
        if getattr(module, "tp_shard_dim", None) is not None:
            for p in module.parameters(recurse=False):
                ids.add(id(p))
    return ids


@torch.no_grad()
def reduce_tp_grads_(model: nn.Module, tp_group) -> None:
    """跨 rank all-reduce「输入各 rank 不同」的 replicated 参数梯度（tp_grad_reduce 标记）。

    q_norm/k_norm 对每头的 q/k 做 RMSNorm：TP 下各 rank 只见自己的头切片，其权重
    梯度是**局部部分和**（各 rank 不同），必须 all-reduce 求和才是真全局梯度——
    对应 Megatron finalize_model_grads.allreduce_layernorm_grads 处理 qk_layernorm
    的机制（记忆知识，未对行号）。其余 replicated 参数（embed/块内 norm/lm_head）
    输入经 f/g 算子全 rank 恒等，梯度天然恒等，不进此路径（进了会 tp× 双计）。
    求和不除 tp_size：部分和的并集 = 全局和。必须先于外置裁剪调用。
    """
    if tp_group is None:
        return
    for module in model.modules():
        if getattr(module, "tp_grad_reduce", False):
            for p in module.parameters(recurse=False):
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=tp_group)


@torch.no_grad()
def clip_grad_norm_tp_(model: nn.Module, optimizer, max_norm: float, tp_group) -> torch.Tensor:
    """TP-aware 外置梯度裁剪（TP 模式 config.gradient_clipping=0，训练循环在边界 step 前调用）。

        global_sq = AR_tp(Σ_local sharded sq) + Σ_local replicated sq

    - sharded 参数各 rank 值不同 → 平方和必须跨 rank all-reduce 才是真全局范数；
    - replicated 参数（embed/norm/lm_head）各 rank 梯度恒等（f/g 算子保证反向后
      dL/dy 全 rank 一致）→ 只计一次；若也进 all-reduce 会 tp× 双计；
    - 剪裁系数全 rank 一致（gnorm 相同），对本地全部梯度统一缩放 → TP 不变量保持。

    返回裁剪前的全局范数（标量张量）。eps 约定与 zero_optimizer.step 的内嵌裁剪一致。
    """
    assert max_norm > 0
    sharded_ids = _tp_sharded_param_ids(model)
    dev = optimizer._flats[0].device
    sharded_sq = torch.zeros((), device=dev, dtype=torch.float32)
    replicated_sq = torch.zeros((), device=dev, dtype=torch.float32)
    for fg in optimizer._flats:
        for p in fg.params:
            if p.grad is None:
                continue
            sq = p.grad.detach().float().pow(2).sum()
            if id(p) in sharded_ids:
                sharded_sq += sq
            else:
                replicated_sq += sq
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(sharded_sq, group=tp_group)
    gnorm = torch.sqrt(sharded_sq + replicated_sq + 1e-12)
    if gnorm.item() > max_norm:
        coef = max_norm / gnorm.item()
        for fg in optimizer._flats:
            for p in fg.params:
                if p.grad is not None:
                    p.grad.mul_(coef)
    return gnorm
