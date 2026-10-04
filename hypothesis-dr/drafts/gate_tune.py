"""I139 gate: base vs tuned-pointwise model graphs in one process (cold private cache): build time, per-graph step time, trial time."""
import importlib.util, sys, time, torch
from pathlib import Path
sys.path.insert(0, ".")
torch.set_num_threads(4); torch.set_num_interop_threads(1)
spec = importlib.util.spec_from_file_location("sub", "drafts/I139_tune.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.evaluate import predict
from benchmark.worker import seed_everything
dev = torch.device("cuda")
data = load_split(Path("data"), train=True); test = load_split(Path("data"), train=False)
def T(): torch.cuda.synchronize(); return time.perf_counter()
for name, flag in (("base", False), ("tuned", True)):
    t = T(); state = m.build(BuildContext(dev, {"tune_pointwise": flag})); print(f"gate {name}: build {T() - t:.0f}s", flush=True)
    x = torch.rand(1536, 3, 32, 32, device=dev).to(memory_format=torch.channels_last); y = torch.randint(0, 100, (1536,), device=dev)
    opt = m.make_optimizer(state.model, state.hyp, 10)
    for size, grad, frozen in ((18, True, False), (28, False, False), (32, False, False), (32, False, True)):
        xs = m.downsample(x, size) if size < 32 else x
        state.model.whiten_bias_grad, state.model.freeze_block1 = grad, frozen
        for _ in range(3): m.train_step(state, opt, xs, y)
        t = T()
        for _ in range(30): m.train_step(state, opt, xs, y)
        print(f"gate {name}: step {size}px grad={grad} frozen={frozen}: {(T() - t) / 30 * 1e3:.2f} ms", flush=True)
    state.model.whiten_bias_grad, state.model.freeze_block1 = True, False
    tt = []
    for trial in range(3):
        seed_everything(trial); t = T(); m.prepare(state, data, trial); mdl = m.train(state); tt.append(T() - t); predict(mdl, test.images, dev, 1024); del mdl
    print(f"gate {name}: trials {[round(v, 3) for v in tt]} mean {sum(tt) / 3:.3f}s", flush=True)
