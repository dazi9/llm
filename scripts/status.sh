#!/usr/bin/env bash
# Quick status of the main training run. Usage (any shell):
#   wsl.exe -d Ubuntu -e bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/status.sh
LOG="$HOME/llm-lab/runs/main/train.log"
CSV="$HOME/llm-lab/runs/main/log.csv"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "=== latest train log ==="
tail -3 "$LOG"
echo "=== eval history ==="
grep "eval" "$LOG" | tail -6
echo "=== ETA ==="
if [ -f "$CSV" ]; then
  python3 - "$CSV" <<'EOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0].isdigit() and r[5]]
step, tok_s = int(rows[-1][0]), float(rows[-1][3])
total = 68000
remain_tok = (total - step) * 16384
remain_h = remain_tok / max(tok_s, 1) / 3600
print(f"step {step}/{total} ({step/total*100:.1f}%) | {tok_s:,.0f} tok/s | ~{remain_h:.1f}h left")
EOF
fi
