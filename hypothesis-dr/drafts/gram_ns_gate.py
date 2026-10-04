"""I087 gate: Gram Newton-Schulz vs the tip's batched NS3 on real momentum matrices (seed 0), and Muon GPU time."""
import importlib.util, sys, torch
from pathlib import Path
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split

def gram_ns(G, steps, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    R = (X @ X.mT).float()
    I = torch.eye(R.size(-1), device=R.device).expand_as(R)
    Q = I
    for _ in range(steps):
        A = Q @ R @ Q.mT
        Q = (a * I + b * A + c * A @ A) @ Q
    return Q.bfloat16() @ X

dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {}))
torch.manual_seed(0); m.prepare(state, data, 0)
errs = {}
import os
SKIP_TRAIN = os.environ.get("SKIP_TRAIN") == "1"
orig = m.Muon.step
count = [0]
def step(self):
    count[0] += 1
    if count[0] in (1, 100, 216):
        with torch.no_grad():
            for idx in self.shape_groups:
                ps = [(self.params[i], self.bufs[i]) for i in idx if self.params[i].grad is not None]
                if not ps: continue
                G = torch.stack([(p.grad + self.momentum * (self.momentum * b + p.grad)).reshape(len(p), -1) for p, b in ps])
                ref = m._newtonschulz(G, self.ns_steps).float(); new = gram_ns(G, self.ns_steps).float()
                errs.setdefault(tuple(G.shape[1:]), []).append(((new - ref).abs().max() / ref.abs().max()).item())
    orig(self)
m.Muon.step = step
m.train(state) if not SKIP_TRAIN else None
for k, v in errs.items():
    print("gate1", k, ["%.2e" % e for e in v])
# timing: NS-only GPU time per step for all groups, ref vs gram
mats = [torch.randn(k, 2 * mm, nn, device=dev) for k, mm, nn in [(1, 64, 216), (1, 64, 1152), (1, 384, 1152), (2, 384, 3456), (1, 576, 3456), (2, 576, 5184)]]
mats = [torch.randn(*shape, device=dev) for shape in [(1, 128, 216), (2, 128, 1152), (1, 384, 1152), (2, 384, 3456), (1, 576, 3456), (2, 576, 5184)]]
fref = torch.compile(lambda Gs: [m._newtonschulz(G, 3) for G in Gs], dynamic=False)
fgram = torch.compile(lambda Gs: [gram_ns(G, 3) for G in Gs], dynamic=False)
for name, f in (("ref", fref), ("gram", fgram)):
    for _ in range(3): f(mats)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20): f(mats)
        torch.cuda.synchronize()
    print(f"gate2 {name} NS GPU time per step: {sum(e.self_device_time_total for e in prof.key_averages())/20/1e3:.3f} ms")
