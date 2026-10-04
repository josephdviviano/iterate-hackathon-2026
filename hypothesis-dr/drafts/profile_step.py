"""Diagnostic (not an experiment): where does a training step's time go?

Usage: CUDA_VISIBLE_DEVICES=3 taskset -c 20-23 uv run python drafts/profile_step.py [submission_dir]
"""

import importlib.util
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.api import BuildContext  # noqa: E402
from benchmark.data import load_split  # noqa: E402

sub = Path(sys.argv[1] if len(sys.argv) > 1 else "submissions/arena") / "submission.py"
spec = importlib.util.spec_from_file_location("sub", sub)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

torch.set_num_threads(4)
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = mod.build(BuildContext(dev, {"epochs": 2}))
torch.manual_seed(0)
mod.prepare(state, data, 0)
torch.cuda.synchronize()
t = time.perf_counter()
mod.train(state)
torch.cuda.synchronize()
print(f"2-epoch train: {time.perf_counter() - t:.3f}s")

# Pure GPU time of fwd+bwd+opt for one batch, vs wall time
x = state.padded[:1024, :, :32, :32].contiguous(memory_format=torch.channels_last)
y = state.labels[:1024]
opt = state.optimizer
for _ in range(5):
    mod.train_step(state, opt, x, y)
torch.cuda.synchronize()
n = 50
t = time.perf_counter()
for _ in range(n):
    mod.train_step(state, opt, x, y)
torch.cuda.synchronize()
print(f"train_step wall: {(time.perf_counter() - t) / n * 1e3:.2f} ms/step")

from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        mod.train_step(state, opt, x, y)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
