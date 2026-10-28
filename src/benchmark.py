"""Throughput benchmark: {fp32, bf16} x {naive, sdpa} x {eager, compile},
plus a forward-only attention microbench (naive vs sdpa vs our Triton kernel).
Writes reports/benchmark.md.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

from model import build_model
from nn_utils import cross_entropy


def time_training_config(preset: str, vocab: int, batch: int, seq: int,
                         precision: str, attn: str, use_compile: bool,
                         warmup: int = 8, timed: int = 25) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(preset, vocab, attn=attn).cuda()
    run = torch.compile(model) if use_compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(precision == "bf16"))
    x = torch.randint(0, vocab, (batch, seq), device="cuda")
    y = torch.randint(0, vocab, (batch, seq), device="cuda")

    last_loss = 0.0

    def step_fn():
        nonlocal last_loss
        opt.zero_grad(set_to_none=True)
        with autocast:
            last_loss = cross_entropy(run(x), y)
        loss = last_loss
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(timed):
        step_fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = batch * seq * timed / dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    res = dict(precision=precision, attn=attn, compile=use_compile,
               tok_s=tok_s, peak_gb=peak, loss=float(last_loss))
    del model, run, opt
    return res


def attention_microbench(b: int = 8, h: int = 12, s: int = 512, d: int = 64,
                         warmup: int = 5, timed: int = 20) -> list[dict]:
    from flash_attention import naive_attention, triton_attention
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    flops = 4 * b * h * s * s * d
    rows = []

    def bench(fn, name):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(timed):
            fn()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / timed * 1e3
        rows.append(dict(name=name, ms=ms, tflops=flops / (ms / 1e3) / 1e12))

    bench(lambda: naive_attention(q, k, v), "naive")
    bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), "sdpa(flash)")
    bench(lambda: triton_attention(q, k, v), "triton(fwd)")
    # correctness of the triton kernel against naive
    out_t = triton_attention(q.float(), k.float(), v.float())
    out_n = naive_attention(q.float(), k.float(), v.float())
    err = (out_t - out_n).abs().max().item()
    rows.append(dict(name="triton max|err| vs naive (fp32)", ms=err, tflops=0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default="reports/benchmark.md")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    configs = [
        ("fp32", "naive", False),
        ("fp32", "sdpa", False),
        ("bf16", "sdpa", False),
        ("bf16", "naive", False),
        ("bf16", "sdpa", True),
    ]
    rows = []
    for precision, attn, comp in configs:
        tag = f"{precision} + {attn} + {'compile' if comp else 'eager'}"
        print(f"[bench] {tag} ...", flush=True)
        try:
            r = time_training_config(args.preset, args.vocab, args.batch, args.seq,
                                     precision, attn, comp)
            rows.append((tag, r))
            print(f"  {r['tok_s']:,.0f} tok/s, peak {r['peak_gb']:.2f}GB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}", flush=True)
            rows.append((tag, None))

    print("[bench] attention microbench ...", flush=True)
    attn_rows = attention_microbench()

    lines = [
        "# Throughput benchmark", "",
        f"- GPU: {torch.cuda.get_device_name(0)}",
        f"- model preset: {args.preset} (build_model), batch {args.batch} x seq {args.seq}, optimizer step included", "",
        "| config | tokens/s | peak VRAM (GB) |",
        "|---|---|---|",
    ]
    for tag, r in rows:
        if r:
            lines.append(f"| {tag} | {r['tok_s']:,.0f} | {r['peak_gb']:.2f} |")
        else:
            lines.append(f"| {tag} | FAILED | - |")
    lines += [
        "", "## Attention forward microbench (B=8, H=12, S=512, D=64, bf16)", "",
        "| impl | ms | TFLOPS |", "|---|---|---|",
    ]
    for r in attn_rows:
        if "err" in r["name"]:
            lines.append(f"| {r['name']} | {r['ms']:.2e} | - |")
        else:
            lines.append(f"| {r['name']} | {r['ms']:.3f} | {r['tflops']:.1f} |")
    report = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
