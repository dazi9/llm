"""Interactive story-completion REPL for a (SFT-)finetuned checkpoint.

  python src/chat.py --ckpt ~/llm-lab/runs/sft/model_final.pt
Type a story opening; the model continues it. Type 'exit' to quit.
"""
from __future__ import annotations

import argparse

import torch

from model import TransformerLM
from sample import generate
from tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    tok = Tokenizer.load(ckpt["tokenizer"])
    model = TransformerLM(vocab_size=config["vocab_size"], d_model=config["d_model"],
                          n_layers=config["n_layers"], n_heads=config["n_heads"],
                          ctx=config["ctx"], d_ff=config["d_ff"], attn="sdpa")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    eos_id = tok._special_to_id.get("<|endoftext|>")

    print("Story completion REPL — type an opening line, 'exit' to quit.")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = generate(model, idx, args.max_new, args.temperature, args.top_p, eos_id)
        print(tok.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
