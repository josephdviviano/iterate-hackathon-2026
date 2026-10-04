"""Diagnostic: CPU wall vs GPU time of Muon.step (ns from argv) inside the real train step."""
import importlib.util, sys, time
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.api import BuildContext
import os; spec = importlib.util.spec_from_file_location("sub", os.environ.get("SUB", "submissions/arena/submission.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ns = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dev = torch.device("cuda")
state = mod.build(BuildContext(dev, {"ns_steps": ns}))
x = torch.rand(1024, 3, 32, 32, device=dev).to(memory_format=torch.channels_last); y = torch.randint(0, 100, (1024,), device=dev)
opts = mod.make_optimizer(state.model, state.hyp, 100)
for _ in range(5): mod.train_step(state, opts, x, y)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): mod.train_step(state, opts, x, y)
torch.cuda.synchronize(); print(f"ns={ns} full train_step wall: {(time.perf_counter()-t)/50*1e3:.3f} ms")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10): opts[1].step()
    torch.cuda.synchronize()
gpu = sum(e.self_device_time_total for e in prof.key_averages()) / 10 / 1e3
print(f"ns={ns} muon.step GPU time: {gpu:.3f} ms")
