#!/usr/bin/env bash
# Environment for the CIFAR-100 task (run by make_workspace.py inside a new workspace):
# the pinned Python environment and the already-downloaded dataset.
set -euo pipefail
ln -s "${CIFAR100_DATA:?set CIFAR100_DATA to the directory that holds the CIFAR-100 download}" data
printf '/data\n' >> .git/info/exclude
uv sync --frozen -q
uv run python -c "import torch; assert torch.__version__.startswith('2.4.0'), torch.__version__"
