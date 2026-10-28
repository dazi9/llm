"""Build (prompt, completion) JSONL pairs from TinyStories for SFT:
prompt = first sentence, completion = the rest. Also emits a small demo file.

  python src/build_sft_pairs.py --data-dir ~/llm-lab/data --n-pairs 60000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-txt", default=os.path.expanduser("~/llm-lab/data/TinyStories-train.txt"))
    ap.add_argument("--out", default=os.path.expanduser("~/llm-lab/data/sft_pairs.jsonl"))
    ap.add_argument("--n-pairs", type=int, default=60000)
    ap.add_argument("--min-completion-chars", type=int, default=80)
    args = ap.parse_args()

    rng = random.Random(0)
    pairs = []
    with open(args.train_txt, encoding="utf-8") as f:
        for line in f:  # stories are one-per-line in this txt
            s = line.strip()
            if len(s) < 150:
                continue
            m = SENT_SPLIT.split(s, maxsplit=1)
            if len(m) != 2:
                continue
            prompt, completion = m[0].strip(), m[1].strip()
            if len(completion) < args.min_completion_chars or len(prompt) < 15:
                continue
            pairs.append({"prompt": prompt, "completion": " " + completion})
            if len(pairs) >= args.n_pairs * 3:
                break
    rng.shuffle(pairs)
    pairs = pairs[: args.n_pairs]
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(pairs):,} pairs -> {args.out}")
    print("example:", json.dumps(pairs[0])[:300])


if __name__ == "__main__":
    main()
