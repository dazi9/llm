"""Aggregate ablation runs into a comparison table.
Reads ~/llm-lab/runs/ablation/{variant}_lr{lr}/log.csv, emits reports/ablation.md.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys

RUNS_DIR = os.path.expanduser("~/llm-lab/runs/ablation")


def read_run(path: str) -> dict:
    evals, toks = [], []
    with open(os.path.join(path, "log.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0] == "step":
                continue
            if row[2]:  # val_loss column
                evals.append((int(row[0]), float(row[2])))
            if row[3]:
                toks.append(float(row[3]))
    return dict(evals=evals, tok_s=max(toks) if toks else 0.0)


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        m = re.match(r".*/(\w+)_lr([\d.e-]+)$", d.replace("\\", "/"))
        if not m or not os.path.exists(os.path.join(d, "log.csv")):
            continue
        name, lr = m.group(1), float(m.group(2))
        r = read_run(d)
        if not r["evals"]:
            continue
        val_by_step = dict(r["evals"])
        rows.append(dict(name=name, lr=lr, final=val_by_step[max(val_by_step)],
                         v1000=val_by_step.get(1000), v2000=val_by_step.get(2000),
                         tok_s=r["tok_s"]))

    lines = ["# Architecture ablation — small preset, 2500 steps x 32K tokens (82M tokens/run),",
             "# batch 64x512, bf16+sdpa+compile, param-aligned (gqa2 via d_ff=1600), seed 0", "",
             "| variant | lr | val@1000 | val@2000 | val@2500 (final) | tok/s |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["final"]):
        lines.append(f"| {r['name']} | {r['lr']:.0e} | {r['v1000'] or '-'} | {r['v2000'] or '-'} | "
                     f"**{r['final']:.4f}** | {r['tok_s']:,.0f} |")
    # best per variant
    lines += ["", "## Best per variant (min final val)", ""]
    best: dict[str, dict] = {}
    for r in rows:
        if r["name"] not in best or r["final"] < best[r["name"]]["final"]:
            best[r["name"]] = r
    for name, r in sorted(best.items(), key=lambda kv: kv[1]["final"]):
        lines.append(f"- **{name}**: {r['final']:.4f} @ lr {r['lr']:.0e}")
    text = "\n".join(lines) + "\n"
    print(text)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "ablation.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
