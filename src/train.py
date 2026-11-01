"""Training loop: presets, bf16 autocast, torch.compile, resume, CSV logging.

Usage (inside WSL venv):
  python src/train.py --preset small --data-dir ~/llm-lab/data --out ~/llm-lab/runs/small
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from data import TokenBin
from model import PRESETS, build_model
from nn_utils import AdamW, cosine_lr_schedule, cross_entropy, gradient_clipping

DEFAULT_LRS = {"tiny": 1e-3, "small": 6e-4, "main": 3e-4}
DEFAULT_STEPS = {"tiny": 300, "small": 3000, "main": 60000}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--data-dir", default=os.path.expanduser("~/llm-lab/data"))
    p.add_argument("--out", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "naive"])
    p.add_argument("--d-ff", type=int, default=None, help="override preset d_ff (param alignment)")
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--optimizer", default="ours", choices=["ours", "fused"],
                   help="ours=hand-written AdamW (per-tensor loop); fused=torch fused AdamW. "
                        "Resume must use the same type (state dict formats differ).")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--train-bin", default=None, help="override tokens_train.bin path")
    p.add_argument("--valid-bin", default=None, help="override tokens_valid.bin path")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json path stored in ckpt")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = "cuda"
    out_dir = args.out or os.path.expanduser(f"~/llm-lab/runs/{args.preset}")
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args.preset, args.vocab_size, attn=args.attn, d_ff=args.d_ff,
                        qk_norm=args.qk_norm, n_kv_heads=args.n_kv_heads,
                        use_rope=not args.no_rope).to(device)
    n_params = model.num_params()
    print(f"[config] {json.dumps({**model.config, 'preset': args.preset, 'n_params': n_params})}", flush=True)

    lr0 = args.lr or DEFAULT_LRS[args.preset]
    if args.optimizer == "fused":
        opt = torch.optim.AdamW(model.parameters(), lr=lr0, betas=(args.beta1, args.beta2),
                                eps=1e-8, weight_decay=args.wd, fused=True)
    else:
        opt = AdamW(model.parameters(), lr=lr0, betas=(args.beta1, args.beta2),
                    weight_decay=args.wd)
    train_bin = TokenBin(args.train_bin or os.path.join(args.data_dir, "tokens_train.bin"))
    val_bin = TokenBin(args.valid_bin or os.path.join(args.data_dir, "tokens_valid.bin"))
    print(f"[data] train tokens: {len(train_bin):,} | valid tokens: {len(val_bin):,}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    max_steps = args.max_steps or DEFAULT_STEPS[args.preset]
    lr_max = args.lr or DEFAULT_LRS[args.preset]
    lr_min = lr_max * args.min_lr_frac

    run_module = model
    if args.compile:
        run_module = torch.compile(model)

    step = 0
    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
        print(f"[resume] from step {step}", flush=True)

    csv_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(csv_path) or step == 0
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_log:
        writer.writerow(["step", "loss", "val_loss", "tok_s", "lr", "mem_gb", "elapsed_s"])
        csv_f.flush()

    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.precision == "bf16"))

    @torch.no_grad()
    def evaluate(nb: int) -> float:
        model.eval()
        losses = []
        vgen = torch.Generator().manual_seed(args.seed + 1234)
        for _ in range(nb):
            x, y = val_bin.get_batch(args.batch_size, args.seq_len, vgen, device)
            with autocast:
                logits = run_module(x)
                losses.append(cross_entropy(logits, y).item())
        model.train()
        return float(np.mean(losses))

    t0 = time.time()
    tokens_seen = step * args.batch_size * args.seq_len * args.accum
    tok_window_start, tok_window_t = tokens_seen, time.time()
    ema_tok_s = None
    model.train()

    if step == 0:
        vl = evaluate(args.eval_batches)
        print(f"[eval] step 0 val_loss {vl:.4f}", flush=True)
        writer.writerow([0, "", f"{vl:.4f}", "", "", "", f"{time.time()-t0:.0f}"])

    while step < max_steps:
        lr = cosine_lr_schedule(step, lr_max, lr_min, args.warmup, max_steps)
        if args.optimizer == "fused":
            opt.param_groups[0]["lr"] = lr
        else:
            opt.lr = lr
        opt.zero_grad()

        step_t0 = time.time()
        loss_acc = 0.0
        for _ in range(args.accum):
            x, y = train_bin.get_batch(args.batch_size, args.seq_len, gen, device)
            with autocast:
                logits = run_module(x)
                loss = cross_entropy(logits, y) / args.accum
            loss.backward()
            loss_acc += loss.item()
        grad_norm = None
        if args.clip:
            gradient_clipping(model.parameters(), args.clip)
            gs = [p.grad.detach().float().pow(2).sum() for p in model.parameters() if p.grad is not None]
            grad_norm = float(torch.sqrt(sum(gs)))
        opt.step()

        step += 1
        tokens_seen += args.batch_size * args.seq_len * args.accum
        if step % args.log_every == 0 or step == max_steps:
            dt = time.time() - tok_window_t
            tok_s = (tokens_seen - tok_window_start) / dt if dt > 0 else 0.0
            ema_tok_s = tok_s if ema_tok_s is None else 0.9 * ema_tok_s + 0.1 * tok_s
            tok_window_start, tok_window_t = tokens_seen, time.time()
            mem_gb = torch.cuda.max_memory_allocated() / 2**30
            print(f"[train] step {step}/{max_steps} loss {loss_acc*args.accum:.4f} "
                  f"tok/s {ema_tok_s:,.0f} lr {lr:.2e} mem {mem_gb:.1f}GB", flush=True)
            writer.writerow([step, f"{loss_acc*args.accum:.4f}", "", f"{ema_tok_s:.0f}",
                             f"{lr:.2e}", f"{mem_gb:.2f}", f"{time.time()-t0:.0f}"])
            csv_f.flush()

        if step % args.eval_every == 0 or step == max_steps:
            vl = evaluate(args.eval_batches)
            print(f"[eval] step {step} val_loss {vl:.4f}", flush=True)
            writer.writerow([step, f"{loss_acc*args.accum:.4f}", f"{vl:.4f}",
                             f"{ema_tok_s or 0:.0f}", f"{lr:.2e}", "", f"{time.time()-t0:.0f}"])
            csv_f.flush()

        if step % args.ckpt_every == 0 or step == max_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(args)}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    torch.save({"model": model.state_dict(), "config": model.config,
                "tokenizer": args.tokenizer or os.path.join(args.data_dir, "tokenizer.json")},
               os.path.join(out_dir, "model_final.pt"))
    print(f"[done] step {step}, {time.time()-t0:.0f}s, final ckpt saved", flush=True)
    csv_f.close()


if __name__ == "__main__":
    main()
