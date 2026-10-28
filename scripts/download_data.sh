#!/usr/bin/env bash
# Download TinyStories corpus via hf-mirror (huggingface.co is unreachable on this network).
set -uo pipefail
cd ~/llm-lab/data
BASE="https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main"
for f in TinyStories-train.txt TinyStories-valid.txt; do
  if [ -s "$f" ]; then echo "=== $f already present ==="; continue; fi
  echo "=== downloading $f $(date '+%F %T') ==="
  curl -L --retry 5 --retry-delay 3 -C - -o "$f.part" "$BASE/$f" && mv "$f.part" "$f"
  if [ ! -s "$f" ]; then echo "DOWNLOAD_FAILED $f"; exit 1; fi
done
wc -c *.txt
echo "DATA_DONE"
