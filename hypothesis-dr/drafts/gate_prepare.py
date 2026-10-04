"""I106 gate: per-component prepare() timing over 3 consecutive trials in one process (mirrors prepare() exactly)."""
import importlib.util, sys, time, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, ".")
torch.set_num_threads(4)
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {}))
def T():
    torch.cuda.synchronize(); return time.perf_counter()
for trial in range(3):
    torch.manual_seed(trial)
    t = {}; t0 = T()
    model = state.model; hyp = state.hyp
    for mod in model.modules():
        if mod is not model and hasattr(mod, "reset_parameters"): mod.reset_parameters()
        if isinstance(mod, torch.nn.BatchNorm2d): mod.reset_running_stats()
    t1 = T(); t["reset"] = t1 - t0
    images = data.images.to(dev, non_blocking=True); t2 = T(); t["h2d"] = t2 - t1
    images = images.float().div_(255); labels = data.labels.to(dev, non_blocking=True); t3 = T(); t["float+labels"] = t3 - t2
    mean = images.mean(dim=(0, 2, 3), keepdim=True); std = images.std(dim=(0, 2, 3), keepdim=True); t4 = T(); t["meanstd"] = t4 - t3
    with torch.no_grad():
        model.mean.copy_(mean); model.std.copy_(std)
        m.init_whitening(model.whiten, ((images[:5000] - mean) / std))
    t5 = T(); t["whiten"] = t5 - t4
    r = hyp["translate"]; padded = F.pad(images, (r, r, r, r), mode="reflect"); t6 = T(); t["pad"] = t6 - t5
    flip = torch.rand(len(images), device=dev) < 0.5; t7 = T(); t["flip"] = t7 - t6
    opt = m.make_optimizer(model, hyp, 216); la = m.Lookahead(model); model.train(); t8 = T(); t["opt+lookahead"] = t8 - t7
    print(f"trial {trial}: total {t8 - t0:.3f}s " + " ".join(f"{k}={v*1e3:.1f}ms" for k, v in t.items()))
