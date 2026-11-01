#!/usr/bin/env bash
# Download 3 Fineweb-Edu sample/10BT shards via hf-mirror for the v2 general-text run.
set -uo pipefail
cd ~/llm-lab/data
mkdir -p fineweb
BASE="https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main"
for f in 000_00000.parquet 000_00001.parquet 000_00002.parquet; do
  if [ -s "fineweb/$f" ]; then echo "=== $f already present ==="; continue; fi
  echo "=== downloading $f $(date '+%F %T') ==="
  curl -sL --retry 5 --retry-delay 3 -C - -o "fineweb/$f.part" "$BASE/sample/10BT/$f" \
    && mv "fineweb/$f.part" "fineweb/$f"
  if [ ! -s "fineweb/$f" ]; then echo "DOWNLOAD_FAILED $f"; exit 1; fi
done
ls -la fineweb/
echo "FINEWEB_DONE"
