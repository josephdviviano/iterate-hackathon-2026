"""Diagnostic: cost of the per-epoch / per-5-step overheads in train()."""
import importlib.util, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.api import BuildContext
from benchmark.data import load_split
sub = Path(sys.argv[1] if len(sys.argv) > 1 else "submissions/arena") / "submission.py"
spec = importlib.util.spec_from_file_location("sub", sub); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_num_threads(4)
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = mod.build(BuildContext(dev, {"epochs": 2}))
torch.manual_seed(0)
def T(name, fn, n=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); print(f"{name}: {(time.perf_counter()-t)/n*1e3:.2f} ms")
torch.cuda.synchronize(); t=time.perf_counter(); mod.prepare(state, data, 0); torch.cuda.synchronize(); print(f"prepare {time.perf_counter()-t:.3f}s")
T("augment", lambda: mod.augment(state.padded, state.flip_bits, 0, 2))
imgs = mod.augment(state.padded, state.flip_bits, 0, 2)
def permute():
    perm = torch.randperm(50000, device=dev); imgs[perm].contiguous(memory_format=torch.channels_last); state.labels[perm]
T("permute", permute)
T("lookahead", lambda: state.lookahead.update(state.model, 0.5), n=50)
