"""TP 版 nano 架构（Megatron 模式：建模代码 ≠ 并行建模代码）。

设计决策（doc/topics/t2_nano_deepspeed.md §2）：
- 叶子算子（RMSNorm / RotaryEmbedding）直接 import 自 nano_model——保证 V7 前向等价的
  差异只来自并行化本身，不混入叶子算子重实现误差；
- 线性层换 Column/Row 并行版本（见 tensor_parallel.py 的 f/g 通信规则）；
- 模块命名与 NanoModel 完全一致（model.layers.{i}.self_attn.q_proj.weight ...），
  因此 C1 导出 = 直接拼回全量 state_dict，无需键名翻译；
- C1 加载：nano 权重文件极小，每 rank 读全量后本地切片（大模型生产做 shard-aware
  加载，此处差异显式记录于 root 记录）；
- GQA 头切分：q/kv 沿头维连续切（tp=2 → rank0 = q0-3/kv0，rank1 = q4-7/kv1），
  本地 group size 不变（n_heads/tp : n_kv/tp = 4:1）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file

from nano_model.config import NanoConfig
from nano_model.norms import RMSNorm
from nano_model.rotary import RotaryEmbedding

from .tensor_parallel import ColumnParallelLinear, RowParallelLinear, tp_rank, tp_world_size


class TPGQAAttention(nn.Module):
    def __init__(self, config: NanoConfig, tp_group) -> None:
        super().__init__()
        tp = tp_world_size(tp_group)
        assert config.n_heads % tp == 0 and config.n_kv_heads % tp == 0, "GQA 头须整除 tp"
        self.n_heads = config.n_heads // tp
        self.n_kv_heads = config.n_kv_heads // tp
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        self.tp_group = tp_group

        self.q_proj = ColumnParallelLinear(config.hidden_size, config.n_heads * config.head_dim, tp_group)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.n_kv_heads * config.head_dim, tp_group)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.n_kv_heads * config.head_dim, tp_group)
        self.o_proj = RowParallelLinear(config.n_heads * config.head_dim, config.hidden_size, tp_group)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # head_dim 不切 → replicated
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        # 输入是各 rank 自己的头切片 → 反向梯度为局部部分和，需跨 TP all-reduce
        #（reduce_tp_grads_，对应 Megatron allreduce_layernorm_grads 对 qk_layernorm 的处理）
        self.q_norm.tp_grad_reduce = True
        self.k_norm.tp_grad_reduce = True
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(B, T, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = self.rotary(positions, q, k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, heads, T, D]
        # 显式 expand 代替 enable_gqa：数值恒等（GQA 即 kv 复制），且不依赖 CPU SDPA 的
        # enable_gqa 后端支持。本地 group size = n_heads//n_kv_heads 与全模型一致。
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


class TPDenseFFN(nn.Module):
    def __init__(self, config: NanoConfig, tp_group) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, tp_group)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, tp_group)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, tp_group)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TPDecoderBlock(nn.Module):
    """pre-norm + 独立 residual，与 nano_model.DecoderBlock 逐行对应。"""

    def __init__(self, config: NanoConfig, tp_group) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TPGQAAttention(config, tp_group)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TPDenseFFN(config, tp_group)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class TPBackbone(nn.Module):
    def __init__(self, config: NanoConfig, tp_group) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # replicated
        self.layers = nn.ModuleList(TPDecoderBlock(config, tp_group) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class TPNanoModel(nn.Module):
    """与 nano_model.NanoModel 接口同构（forward → hidden / compute_logits → logits），
    训练循环无感切换。tp=1 时全部退化为 replicated，代码路径与 tp>1 统一。"""

    def __init__(self, config: NanoConfig, tp_group) -> None:
        super().__init__()
        assert config.tie_word_embeddings, "C1 约定 tied lm_head"
        self.config = config
        self.model = TPBackbone(config, tp_group)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # tied（与 NanoModel 相同）
        self.tp_group = tp_group

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def _slice_shard(full: torch.Tensor, shard_dim: int, rank: int, tp: int) -> torch.Tensor:
    n = full.shape[shard_dim] // tp
    s = rank * n
    return (full[s:s + n, :] if shard_dim == 0 else full[:, s:s + n]).contiguous()


def load_tp_model(ckpt_dir: str | Path, tp_group) -> TPNanoModel:
    """从 C1 目录（config.json + model.safetensors）构建 TP 模型并按 rank 切片加载。

    切片规则：Column（tp_shard_dim=0）取行分片，Row（tp_shard_dim=1）取列分片，
    replicated 取全量。权重来自 safetensors 张量视图（contiguous 复制进参数）。
    """
    ckpt_dir = Path(ckpt_dir)
    config = NanoConfig.load(ckpt_dir / "config.json")
    model = TPNanoModel(config, tp_group)
    rank, tp = tp_rank(tp_group), tp_world_size(tp_group)
    full = load_file(str(ckpt_dir / "model.safetensors"))

    param_by_name = dict(model.named_parameters())
    for name in full:
        assert name in param_by_name, f"C1 key 不在 TP 模型中：{name}"
    # 走 module 遍历（param 自身不挂标记，标记在宿主 Linear 上）
    shard_dim_of = {}
    for module_name, module in model.named_modules():
        shard = getattr(module, "tp_shard_dim", None)
        if shard is not None:
            for pname, p in module.named_parameters(recurse=False):
                shard_dim_of[f"{module_name}.{pname}" if module_name else pname] = shard

    with torch.no_grad():
        for name, p in param_by_name.items():
            if name not in full:  # tied lm_head 不在 C1 文件里，随 embed 加载
                assert p is param_by_name["model.embed_tokens.weight"], f"unexpected missing key {name}"
                continue
            src = full[name]
            shard_dim = shard_dim_of.get(name)
            p.copy_(_slice_shard(src, shard_dim, rank, tp) if shard_dim is not None else src)
    return model


@torch.no_grad()
def export_full_state_dict(model: TPNanoModel) -> dict[str, torch.Tensor]:
    """按 shard 标记 all-gather 拼回全量 state_dict（键名 = C1，rank 0 可直接 save_file）。

    每 rank 都返回全量字典（nano 尺度代价可忽略）；生产实现只在 rank 0 组装。
    """
    tp = tp_world_size(model.tp_group)
    if tp == 1:
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    shard_dim_of = {}
    for module_name, module in model.named_modules():
        shard = getattr(module, "tp_shard_dim", None)
        if shard is not None:
            for pname, p in module.named_parameters(recurse=False):
                shard_dim_of[f"{module_name}.{pname}" if module_name else pname] = shard

    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        shard_dim = shard_dim_of.get(name)
        if shard_dim is None:
            out[name] = p.detach().clone()
            continue
        parts = [torch.empty_like(p) for _ in range(tp)]
        dist.all_gather(parts, p.detach().contiguous(), group=model.tp_group)
        full = torch.cat(parts, dim=shard_dim)
        out[name] = full
    return out
