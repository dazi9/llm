"""Optimizer-step microbenchmark: our hand-written AdamW vs torch AdamW
(single-tensor default and fused), measuring pure optimizer.step() time.

Runs on the `small` preset (VRAM-friendly while training occupies the GPU);
scales ~linearly to the 100M model. Appends results to reports/benchmark.md.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "src")
from model import build_model  # noqa: E402
from nn_utils import AdamW  # noqa: E402


def bench(step_fn, warmup=20, timed=200) -> float:
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(timed):
        step_fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / timed * 1e3  # ms per step


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    model = build_model("small", vocab_size=10000).cuda()
    n_params = model.num_params()
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    ours = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    ref_single = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    ref_fused = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                                  weight_decay=0.01, fused=True)

    rows = []
    rows.append(("ours (hand-written, python loop)", bench(lambda: ours.step())))
    rows.append(("torch AdamW (single-tensor)", bench(lambda: ref_single.step())))
    rows.append(("torch AdamW (fused=True)", bench(lambda: ref_fused.step())))

    lines = [
        "", "## Optimizer step microbenchmark (pure step, no fwd/bwd)", "",
        f"- model: small preset, {n_params:,} params (28.9M); scales ~linearly to 100M",
        "- hand-written AdamW loops over ~200 tensors in Python; fused does it in one kernel", "",
        "| implementation | ms/step |",
        "|---|---|",
    ]
    for name, ms in rows:
        lines.append(f"| {name} | {ms:.3f} |")
    lines += [
        "", f"- extrapolated to 100M params: ours ≈ {rows[0][1]*3.47:.2f} ms/step, "
        f"fused ≈ {rows[2][1]*3.47:.2f} ms/step",
        f"- at 58.7K tok/s the training step itself is ~{16384/58700*1e3:.0f} ms "
        "(16384 tokens/step), so the optimizer share is "
        f"{rows[0][1]*3.47/(16384/58700*1e3)*100:.1f}% (ours) vs "
        f"{rows[2][1]*3.47/(16384/58700*1e3)*100:.1f}% (fused) of step time",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    with open("reports/benchmark.md", "a", encoding="utf-8") as f:
        f.write(text)
    print("ADAMW_BENCH_DONE")


if __name__ == "__main__":
    main()
