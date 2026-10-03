#!/usr/bin/env bash
# Create the local accuracy-only dev stack (.venv-blackwell) for GPUs the pinned torch 2.4.0
# cannot run (sm_100/sm_120). Timing evidence never comes from this stack.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv --python 3.12 .venv-blackwell
VIRTUAL_ENV=$PWD/.venv-blackwell uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -r research/env/blackwell-freeze.txt
VIRTUAL_ENV=$PWD/.venv-blackwell uv pip install --no-deps -e .
.venv-blackwell/bin/python -m benchmark.data --root data
.venv-blackwell/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
