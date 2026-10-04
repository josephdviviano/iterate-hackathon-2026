"""I083 calibration: ||NS3(X/||X||)||_F / sqrt(min(m,n)) per shape group on the tip's real momentum matrices."""
import importlib.util, sys, torch
from pathlib import Path
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {}))
torch.manual_seed(0); m.prepare(state, data, 0)
mu = state.optimizer[1]
ratios = {}
orig = m.Muon.step
def step(self):
    with torch.no_grad():
        for idx in self.shape_groups:
            ps = [self.params[i] for i in idx if self.params[i].grad is not None]
            if not ps: continue
            G = torch.stack([(p.grad + self.momentum * (self.momentum * b + p.grad)).reshape(len(p), -1) for p, b in zip(ps, [self.bufs[i] for i in idx])])
            U = m._newtonschulz(G, self.ns_steps).float()
            r = (U.norm(dim=(1, 2)) / min(G.shape[1:]) ** 0.5).mean().item()
            ratios.setdefault(tuple(G.shape[1:]), []).append(r)
    orig(self)
m.Muon.step = step
m.train(state)
for k, v in ratios.items():
    print(k, "steps", len(v), "ratio first %.3f mid %.3f last %.3f mean %.3f" % (v[0], v[len(v)//2], v[-1], sum(v)/len(v)))
