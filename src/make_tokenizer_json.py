"""Convert our hand-written byte-level BPE into a GPT-2-style HF tokenizer.json,
which convert_hf_to_llama.py understands (BPE vocab + merges + GPT-2 regex
pretokenizer — the exact same scheme our Tokenizer implements).

  python src/make_tokenizer_json.py --ckpt ~/llm-lab/runs/main/model_final.pt \
      --out ~/llm-lab/export/v1
"""
from __future__ import annotations

import argparse
import json
import os

from tokenizer import Tokenizer
import tokenizer as tokenizer_mod


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte->unicode char mapping (printable bytes map to themselves)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = Tokenizer.load(ckpt["tokenizer"] if isinstance(ckpt["tokenizer"], str)
                         else ckpt["tokenizer"])
    b2u = bytes_to_unicode()

    def u(bs: bytes) -> str:
        return "".join(b2u[b] for b in bs)

    vocab: dict[str, int] = {}
    for i in range(256):
        vocab[u(bytes([i]))] = i
    for rank, (a, b) in enumerate(tok.merges):
        merged = a + b
        vocab[u(merged)] = 256 + rank
    specials = {}
    for s, sid in tok._special_to_id.items():
        vocab[s] = sid
        specials[s] = sid

    merges_list = [f"{u(a)} {u(b)}" for a, b in tok.merges]

    tj = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": sid, "content": s, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for s, sid in specials.items()
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": tokenizer_mod._PRETOKEN_RE.pattern},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True,
                           "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False,
            "vocab": vocab,
            "merges": merges_list,
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)
    with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast",
                   "eos_token": "<|endoftext|>", "bos_token": "<|endoftext|>",
                   "model_max_length": int(1e30)}, f)

    # self-check: round-trip must match our own tokenizer exactly
    test = "Once upon a time, there was a girl named Lily! She ran 42 miles."
    ids = tok.encode(test)
    print(f"vocab {tok.vocab_size}, merges {len(tok.merges)} -> tokenizer.json written")
    print(f"roundtrip ids[:8] = {ids[:8]}")
    assert tok.decode(ids) == test


if __name__ == "__main__":
    main()
