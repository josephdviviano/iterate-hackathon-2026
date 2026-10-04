"""I107 gate: GPU time of per-epoch data work (augment+jitter, permute/gather) and per-batch downsample in one E069 trial."""
import importlib.util, sys, time, torch
from pathlib import Path
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {}))
torch.manual_seed(0); m.prepare(state, data, 0)
hyp = state.hyp; r = hyp["translate"]; n = 50000; bs = hyp["batch_size"]
def T(): torch.cuda.synchronize(); return time.perf_counter()
aug = perm = down = 0.0
for epoch in range(7):
    t0 = T()
    imgs = m.augment(state.padded, state.flip_bits, epoch, r, hyp["contrast"], hyp["brightness"], epoch + 1 > hyp["res_schedule"][-1][0])
    t1 = T()
    p = torch.randperm(n, device=dev); imgs = imgs[p].contiguous(memory_format=torch.channels_last); labels = state.labels[p]
    t2 = T(); aug += t1 - t0; perm += t2 - t1
total_steps = int((n // bs) * hyp["epochs"])
lowres = [(int((n // bs) * u), s) for u, s in hyp["res_schedule"]]
xb = imgs[:bs]
for step in range(total_steps):
    size = next((s for u, s in lowres if step < u), 32)
    if size < 32:
        t0 = T(); m.downsample(xb, size); down += T() - t0
print(f"gate: augment {aug:.3f}s, permute/gather {perm:.3f}s, downsample {down:.3f}s, total {aug+perm+down:.3f}s per trial")
