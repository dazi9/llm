"""Sample from a trained checkpoint.

  python src/sample.py --ckpt ~/llm-lab/runs/small/model_final.pt \
      --prompt "Once upon a time" --max-new 200
"""
from __future__ import annotations

import argparse
import os

import torch

from model import TransformerLM
from tokenizer import Tokenizer


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new: int, temperature: float, top_p: float,
             eos_id: int | None) -> torch.Tensor:
    model.eval()
    ctx = model.config["ctx"]
    for _ in range(max_new):
        idx_cond = idx[:, -ctx:]
        logits = model(idx_cond)[:, -1].float()
        logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum - sorted_probs < top_p  # keep tokens up to and crossing p
            remove = torch.zeros_like(keep).scatter(-1, sorted_idx, ~keep)
            logits = logits.masked_fill(remove, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="number of samples")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = ckpt["config"]
    else:
        # periodic checkpoint: rebuild from training args
        from model import PRESETS
        a = ckpt["args"]
        config = {**PRESETS[a["preset"]], "vocab_size": a["vocab_size"],
                  "tie_embeddings": False}
    tok_path = ckpt.get("tokenizer") or os.path.join(ckpt["args"]["data_dir"], "tokenizer.json")
    tok = Tokenizer.load(tok_path)

    model = TransformerLM(vocab_size=config["vocab_size"], d_model=config["d_model"],
                          n_layers=config["n_layers"], n_heads=config["n_heads"],
                          ctx=config["ctx"], d_ff=config.get("d_ff"), attn="sdpa",
                          qk_norm=config.get("qk_norm", False),
                          n_kv_heads=config.get("n_kv_heads"),
                          use_rope=config.get("use_rope", True))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    eos_id = tok._special_to_id.get("<|endoftext|>")
    for i in range(args.n):
        idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
        out = generate(model, idx, args.max_new, args.temperature, args.top_p, eos_id)
        text = tok.decode(out[0].tolist())
        print(f"===== sample {i+1} =====")
        print(text)
        print()


if __name__ == "__main__":
    main()
