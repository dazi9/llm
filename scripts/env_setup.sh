#!/usr/bin/env bash
# One-shot environment bootstrap inside WSL Ubuntu.
# Idempotent: safe to re-run if a step already succeeded.
set -uo pipefail
echo "=== env_setup start $(date '+%F %T') ==="

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "UV_INSTALL_FAILED"; exit 1; }
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version || { echo "UV_FAILED"; exit 1; }

mkdir -p ~/llm-lab/data ~/llm-lab/runs ~/llm-lab/cache
cd ~/llm-lab

if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python 3.13 || { echo "VENV_FAILED"; exit 1; }
fi
source .venv/bin/activate

if ! python -c "import torch" >/dev/null 2>&1; then
  # Pin 2.9.1: PyPI default variant is cu12.8 (sm_120 OK, driver 572.84 OK).
  # Newer torch resolves to cu13 deps which need driver >= 580 -> fails on this box.
  uv pip install "torch==2.9.1" --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    || uv pip install "torch==2.9.1" --index-url https://mirrors.aliyun.com/pypi/simple/ \
    || { echo "TORCH_FAILED"; exit 1; }
fi

if ! python -c "import numpy, tiktoken, huggingface_hub, pytest" >/dev/null 2>&1; then
  uv pip install numpy tiktoken huggingface_hub pytest tqdm \
    || uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple numpy tiktoken huggingface_hub pytest tqdm \
    || { echo "DEPS_FAILED"; exit 1; }
fi

python - <<'EOF'
import sys, time
import torch
print("torch", torch.__version__, "| cuda runtime", torch.version.cuda, "| available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("NO_CUDA")
    sys.exit(2)
print("device:", torch.cuda.get_device_name(0), "| compute capability:", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(10):
    y = x @ x
torch.cuda.synchronize()
tflops = 10 * 2 * 4096**3 / (time.time() - t) / 1e12
print(f"bf16 matmul OK, ~{tflops:.0f} TFLOPS")
import triton
print("triton", triton.__version__)
EOF
rc=$?
echo "=== env_setup done rc=$rc $(date '+%F %T') ==="
exit $rc
