"""Controlled architecture ablation driver.

Variants (all on `small` preset, param-aligned via d_ff adjustment):
  baseline / qk_norm / gqa2 (n_kv_heads=2) / nope (no positional encoding)
Each runs 2500 steps at batch 64x512 with 3 learning rates, seed fixed.
Usage:  python src/run_ablation.py            # runs everything sequentially
        python src/run_ablation.py --plan     # just print the plan (params, d_ff)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import build_model  # noqa: E402

VARIANTS: dict[str, dict] = {
    "baseline": dict(),
    "qk_norm": dict(qk_norm=True),
    "gqa2": dict(n_kv_heads=2),
    "nope": dict(use_rope=False),
}
LRS = [3e-4, 6e-4, 1.2e-3]
STEPS = 2500
BATCH = 64


def count_params(variant: dict, d_ff: int | None) -> int:
    m = build_model("small", vocab_size=10000, d_ff=d_ff, **variant)
    n = m.num_params()
    del m
    return n


def align_d_ff(variant: dict, base_d_ff: int = 1344) -> tuple[int, int, int]:
    """Find d_ff (multiple of 64) minimizing the param gap vs baseline. Returns (d_ff, params, gap)."""
    base_n = count_params({}, base_d_ff)
    best = (base_d_ff, count_params(variant, base_d_ff), abs(count_params(variant, base_d_ff) - base_n))
    for d_ff in range(base_d_ff - 384, base_d_ff + 641, 64):
        n = count_params(variant, d_ff)
        if abs(n - base_n) < best[2]:
            best = (d_ff, n, abs(n - base_n))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    args = ap.parse_args()

    plan = {}
    for name, variant in VARIANTS.items():
        d_ff, n, gap = align_d_ff(variant)
        plan[name] = dict(variant=variant, d_ff=d_ff, params=n, gap=gap)
        print(f"[plan] {name:10s} d_ff={d_ff} params={n:,} (gap {gap:,})")

    if args.plan:
        return

    names = args.only.split(",") if args.only else list(VARIANTS)
    for name in names:
        cfg = plan[name]
        for lr in LRS:
            out = os.path.expanduser(f"~/llm-lab/runs/ablation/{name}_lr{lr:.0e}")
            if os.path.exists(os.path.join(out, "model_final.pt")):
                print(f"[skip] {name} lr={lr} (model_final.pt exists)", flush=True)
                continue
            cmd = [sys.executable, "src/train.py", "--preset", "small",
                   "--compile", "--batch-size", str(BATCH), "--max-steps", str(STEPS),
                   "--warmup", "100", "--lr", str(lr), "--eval-every", "500",
                   "--ckpt-every", "100000", "--log-every", "100",
                   "--out", out, "--d-ff", str(cfg["d_ff"])]
            for k, v in cfg["variant"].items():
                if v is True:
                    cmd.append("--" + k.replace("_", "-"))
                elif v is not None:
                    cmd += ["--" + k.replace("_", "-"), str(v)]
            print(f"[run] {name} lr={lr} -> {out}", flush=True)
            r = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), ".."))
            if r.returncode != 0:
                print(f"[FAIL] {name} lr={lr} rc={r.returncode}", flush=True)
    print("ABLATION_BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
