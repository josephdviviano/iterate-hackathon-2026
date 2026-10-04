"""I131 gate: per-component prepare() timing over 3 harness-like trials (prepare, train, predict) on the I125 base."""
import importlib.util, sys, time, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, ".")
torch.set_num_threads(4); torch.set_num_interop_threads(1)
spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else "drafts/I125_base.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.evaluate import predict
from benchmark.worker import seed_everything
dev = torch.device("cuda")
data = load_split(Path("data"), train=True); test = load_split(Path("data"), train=False)
state = m.build(BuildContext(dev, {}))
def T(): torch.cuda.synchronize(); return time.perf_counter()
for trial in range(3):
    seed_everything(trial)
    t = {}; t0 = T(); model = state.model; hyp = state.hyp
    for mod in model.modules():
        if mod is not model and hasattr(mod, "reset_parameters"): mod.reset_parameters()
        if isinstance(mod, torch.nn.BatchNorm2d): mod.reset_running_stats()
    t1 = T(); t["reset"] = t1 - t0
    images = data.images.to(dev, non_blocking=True); t2 = T(); t["h2d"] = t2 - t1
    images = images.float().div_(255); labels = data.labels.to(dev, non_blocking=True); t3 = T(); t["float"] = t3 - t2
    mean = images.mean(dim=(0, 2, 3), keepdim=True); std = images.std(dim=(0, 2, 3), keepdim=True); t4 = T(); t["meanstd"] = t4 - t3
    with torch.no_grad():
        model.mean.copy_(mean); model.std.copy_(std); m.init_whitening(model.whiten, (images[:5000] - mean) / std)
    t5 = T(); t["whiten"] = t5 - t4
    r = hyp["translate"]; state.padded = F.pad(images, (r, r, r, r), mode="reflect"); t6 = T(); t["pad"] = t6 - t5
    state.labels = labels; state.flip_bits = torch.rand(len(images), device=dev) < 0.5; t7 = T(); t["flip"] = t7 - t6
    steps_per_epoch = len(images) // hyp["batch_size"]; state.total_steps = int(steps_per_epoch * hyp["epochs"])
    state.optimizer = m.make_optimizer(model, hyp, state.total_steps); state.lookahead = m.Lookahead(model); model.train(); t8 = T(); t["opt"] = t8 - t7
    print(f"trial {trial}: prepare {t8 - t0:.3f}s " + " ".join(f"{k}={v*1e3:.1f}" for k, v in t.items()), flush=True)
    mdl = m.train(state); predict(mdl, test.images, dev, 1024); del mdl
