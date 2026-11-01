#!/usr/bin/env bash
# One-shot morning finalization: sample the finished main model, run SFT,
# sample the SFT model, and copy artifacts to the Windows workspace.
# Run (from Windows, any shell):
#   wsl.exe -d Ubuntu -e bash -lc "bash /mnt/c/Users/dazi9/.zcode/workspace/default/llm-lab/scripts/finalize.sh"
set -e
source ~/llm-lab/.venv/bin/activate
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO"

echo "=== 1) sample main (final) ==="
python src/sample.py --ckpt ~/llm-lab/runs/main/model_final.pt \
  --prompt "Once upon a time" --max-new 250 --n 3 | tee reports/main-samples.txt

echo "=== 2) SFT 800 steps (~3 min on GPU) ==="
python src/sft.py --init ~/llm-lab/runs/main/model_final.pt \
  --pairs ~/llm-lab/data/sft_pairs.jsonl --steps 800 --out ~/llm-lab/runs/sft

echo "=== 3) sample SFT model ==="
python src/sample.py --ckpt ~/llm-lab/runs/sft/model_final.pt \
  --prompt "Once upon a time, there was a little boy named Sam." --max-new 200 --n 2 | tee reports/sft-samples.txt

echo "=== 4) copy artifacts to workspace ==="
mkdir -p runs-main
cp ~/llm-lab/runs/main/model_final.pt runs-main/ || true
cp ~/llm-lab/runs/main/log.csv runs-main/ || true
cp ~/llm-lab/runs/main/train.log runs-main/ || true

echo "FINALIZE_DONE"
