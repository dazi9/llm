#!/usr/bin/env bash
# Fineweb shard download WITH size validation and retry loop.
# hf-mirror sometimes returns a 15-byte "Entry not found" body with rc=0.
set -uo pipefail
cd ~/llm-lab/data/fineweb
BASE="https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main"

download_validated() {
  local f="$1" tries=0
  while [ $tries -lt 8 ]; do
    tries=$((tries+1))
    local expected
    expected=$(curl -sIL --max-time 30 "$BASE/sample/10BT/$f" | grep -i "^content-length" | tail -1 | tr -dc 0-9)
    if [ -z "$expected" ] || [ "$expected" -lt 1000000000 ]; then
      echo "[$f] bad/missing content-length ('$expected'), retry $tries"; sleep 10; continue
    fi
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$expected" ]; then
      echo "[$f] already complete"; return 0
    fi
    echo "[$f] downloading (expect $expected bytes, try $tries) $(date '+%T')"
    curl -sL --retry 5 --retry-delay 5 -C - -o "$f.part" "$BASE/sample/10BT/$f"
    local actual; actual=$(stat -c%s "$f.part" 2>/dev/null || echo 0)
    if [ "$actual" -eq "$expected" ]; then
      mv "$f.part" "$f"; echo "[$f] OK ($actual bytes)"; return 0
    fi
    echo "[$f] size mismatch (got $actual, want $expected), retrying with resume"
    sleep 5
  done
  echo "[$f] FAILED after $tries tries"; return 1
}

for f in 001_00000.parquet 002_00000.parquet; do
  download_validated "$f" || exit 1
done
echo "FINEWEB_DONE"
