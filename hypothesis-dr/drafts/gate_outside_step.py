"""I122 gate: wall-time breakdown of one E081 trial: train_step vs lookahead vs per-epoch augment vs everything else."""
import importlib.util, sys, time, torch
from pathlib import Path
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else "drafts/E081_bundle_submission.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
torch.set_num_threads(4)
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {}))
acc = {"step": 0.0, "lookahead": 0.0, "augment": 0.0, "resize": 0.0}
def wrap(name, fn):
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize(); acc[name] += time.perf_counter() - t; return r
    return w
m.train_step = wrap("step", m.train_step)
m.Lookahead.update = wrap("lookahead", m.Lookahead.update)
m.augment_and_permute = wrap("augment", m.augment_and_permute)
m.downsample = wrap("resize", m.downsample); acc["resize"] = 0.0
for seed in (0, 1):
    for k in acc: acc[k] = 0.0
    torch.manual_seed(seed); torch.cuda.synchronize(); t0 = time.perf_counter(); m.prepare(state, data, seed); t1 = time.perf_counter()
    m.train(state); torch.cuda.synchronize(); t2 = time.perf_counter()
    other = (t2 - t1) - sum(acc.values())
    print(f"gate seed {seed}: prepare {t1-t0:.3f}s train {t2-t1:.3f}s | " + " ".join(f"{k} {v:.3f}" for k, v in acc.items()) + f" other {other:.3f}")
