"""V10 train→serve 交接 smoke：save_checkpoint 产物 → NanoModel → greedy 生成。

用法：python smoke_generate.py --ckpt <tag_dir> [--prompt "First Citizen:\n"] [--n 200]

ckpt 目录布局（见 nano_deepspeed/checkpoint.py）：
    model/        C1 导出（model.safetensors + config.json）——serve 侧唯一消费物
    vocab.json    tokenizer（save_checkpoint 写入）

greedy 解码：每步取 argmax；上下文窗口截到 C1 max_position_embeddings（RoPE 缓存上限）。
"""

from __future__ import annotations

import argparse

import torch

from nano_model.model import NanoModel
from nano_model.tokenizer import CharTokenizer


@torch.no_grad()
def generate(ckpt_dir: str, prompt: str = "First Citizen:\n", n: int = 200) -> str:
    tok = CharTokenizer.load(f"{ckpt_dir}/vocab.json")
    model = NanoModel.load_pretrained(f"{ckpt_dir}/model").eval()
    max_pos = model.config.max_position_embeddings
    ids = tok.encode(prompt)
    for _ in range(n):
        ctx = ids[-max_pos:]
        x = torch.tensor([ctx])
        pos = torch.arange(len(ctx)).unsqueeze(0)
        logits = model.compute_logits(model(x, pos))
        ids.append(int(logits[0, -1].argmax()))
    return tok.decode(ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="save_checkpoint 的 tag 目录")
    p.add_argument("--prompt", default="First Citizen:\n")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    print(generate(args.ckpt, args.prompt, args.n))


if __name__ == "__main__":
    main()
