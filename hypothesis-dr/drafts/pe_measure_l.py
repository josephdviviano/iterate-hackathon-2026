"""I089: distribution of normalized singular values sigma_i/||X||_F of the tip's real Muon inputs (seed 0, steps 1/100/216)."""
import importlib.util, sys, torch
from pathlib import Path
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {}))
torch.manual_seed(0); m.prepare(state, data, 0)
svs, mats = [], []
orig = m.Muon.step
count = [0]
def step(self):
    count[0] += 1
    if count[0] in (1, 100, 216):
        with torch.no_grad():
            for idx in self.shape_groups:
                for i in idx:
                    p, b = self.params[i], self.bufs[i]
                    if p.grad is None: continue
                    G = (p.grad + self.momentum * (self.momentum * b + p.grad)).reshape(len(p), -1).float()
                    s = torch.linalg.svdvals(G / G.norm())
                    svs.append(s.cpu()); mats.append(G.cpu())
    orig(self)
m.Muon.step = step
m.train(state)
allsv = torch.cat(svs)
print("count", len(svs), "sv quantiles (frob-normalised): p1 %.2e p5 %.2e p50 %.2e max %.2e" % tuple(torch.quantile(allsv, torch.tensor([0.01, 0.05, 0.5, 1.0])).tolist()))
torch.save(mats, "/tmp/pe_mats.pt")
