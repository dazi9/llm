"""SFT: finetune a pretrained checkpoint on (prompt, completion) JSONL pairs.
Prompt tokens are masked out of the loss (-100). Produces a chat-able model.

  python src/sft.py --init ~/llm-lab/runs/main/model_final.pt \
      --pairs data/sft_pairs.jsonl --out ~/llm-lab/runs/sft
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from model import TransformerLM
from nn_utils import AdamW, cosine_lr_schedule, cross_entropy, gradient_clipping
from tokenizer import Tokenizer


def load_pairs(path: str, tok: Tokenizer, seq_len: int) -> list[tuple[list[int], list[int]]]:
    """Returns list of (input_ids, target_ids) with prompt positions masked."""
    eos = tok._special_to_id.get("<|endoftext|>")
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p_ids = tok.encode(rec["prompt"], allow_special=False)
            c_ids = tok.encode(rec["completion"], allow_special=False)
            ids = p_ids + c_ids + ([eos] if eos else [])
            ids = ids[: seq_len + 1]
            if len(ids) < 2:
                continue
            x = ids[:-1]
            y = ids[1:]
            mask_len = max(0, min(len(p_ids) - 1, len(y)))
            for i in range(mask_len):
                y[i] = -100
            pairs.append((x, y))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/llm-lab/runs/sft"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    config["attn"] = "sdpa"
    tok = Tokenizer.load(ckpt["tokenizer"])
    model = TransformerLM(vocab_size=config["vocab_size"], d_model=config["d_model"],
                          n_layers=config["n_layers"], n_heads=config["n_heads"],
                          ctx=config["ctx"], d_ff=config.get("d_ff"), attn="sdpa",
                          qk_norm=config.get("qk_norm", False),
                          n_kv_heads=config.get("n_kv_heads"),
                          use_rope=config.get("use_rope", True))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    pairs = load_pairs(args.pairs, tok, config["ctx"])
    print(f"[sft] {len(pairs):,} pairs, model {model.num_params():,} params", flush=True)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.lr = cosine_lr_schedule(step, args.lr, args.lr * 0.1, args.warmup, args.steps)
        batch = [pairs[i] for i in torch.randint(len(pairs), (args.batch,), generator=gen).tolist()]
        L = max(len(x) for x, _ in batch)
        x = torch.full((len(batch), L), 0, dtype=torch.long)
        y = torch.full((len(batch), L), -100, dtype=torch.long)
        for j, (xb, yb) in enumerate(batch):
            x[j, : len(xb)] = torch.tensor(xb)
            y[j, : len(yb)] = torch.tensor(yb)
        x, y = x.to(device), y.to(device)
        with autocast:
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                 ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps:
            print(f"[sft] step {step}/{args.steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "config": config, "tokenizer": ckpt["tokenizer"],
                "sft_pairs": os.path.abspath(args.pairs)},
               os.path.join(args.out, "model_final.pt"))
    print("SFT_DONE", flush=True)


if __name__ == "__main__":
    main()
