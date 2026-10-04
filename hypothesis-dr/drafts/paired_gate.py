"""Paired offline gate in ONE process: one build, then for each seed run every config back to back.

Same seeding, prepare/train and test evaluation as the harness worker (benchmark.worker / benchmark.evaluate).
Sharing one build means both configs use the same autotuned kernels, so process-level re-draws cancel.

Usage: python drafts/paired_gate.py SUBMISSION.py SEED_FROM SEED_TO 'BUILD_PARAMS_JSON' 'CONFIG_A_JSON' 'CONFIG_B_JSON' ...
  BUILD_PARAMS go to build(); each CONFIG is a dict of hyp overrides applied before prepare() (must only use shapes that
  build() warmed; extra warmup sizes can be requested with BUILD_PARAMS {"_extra_warm_sizes": [20, ...]}).
"""

import importlib.util
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.api import BuildContext  # noqa: E402
from benchmark.data import load_split  # noqa: E402
from benchmark.evaluate import accuracy, predict  # noqa: E402
from benchmark.worker import seed_everything  # noqa: E402

sub, s0, s1, build_params, *configs = sys.argv[1:]
spec = importlib.util.spec_from_file_location("sub", sub)
m = importlib.util.module_from_spec(spec)
sys.modules["sub"] = m
spec.loader.exec_module(m)
torch.set_num_threads(4)
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # configs may recompile (e.g. model constants); avoid eager fallback
torch.set_num_interop_threads(1)
dev = torch.device("cuda")
train = load_split(Path("data"), train=True)
test = load_split(Path("data"), train=False)
bp = json.loads(build_params)
extra = bp.pop("_extra_warm_sizes", [])
t = time.perf_counter()
state = m.build(BuildContext(dev, bp))
# optional extra warmups (other first-stage resolutions), with and without whitening-bias grad
if extra:
    bs = state.hyp["batch_size"]
    x = torch.rand(bs, 3, 32, 32, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, 100, (bs,), device=dev)
    opt = m.make_optimizer(state.model, state.hyp, 10)
    for size in extra:
        xs = m.downsample(x, size)
        for g in (True, False):
            state.model.whiten_bias_grad = g
            for _ in range(3):
                m.train_step(state, opt, xs, y)
    state.model.whiten_bias_grad = True
torch.cuda.synchronize()
print(f"build {time.perf_counter() - t:.0f}s", flush=True)
base_hyp = dict(state.hyp)
configs = [json.loads(c) for c in configs]
res = {i: {} for i in range(len(configs))}
for seed in range(int(s0), int(s1) + 1):
    for i, cfg in enumerate(configs):
        state.hyp = {**base_hyp, **{k: v for k, v in cfg.items() if not k.startswith("_")}}
        for k, v in cfg.items():  # "_model_<attr>": set a model attribute; "_all_<attr>": set it on every submodule
            if k.startswith("_model_"):
                setattr(state.model, k[len("_model_"):], v)
            if k.startswith("_all_"):
                for mod in state.model.modules():
                    setattr(mod, k[len("_all_"):], v)
        seed_everything(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.prepare(state, train, seed)
        model = m.train(state)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        preds, _ = predict(model, test.images, dev, 1024)
        res[i][seed] = (accuracy(preds, test.labels), dt)
    print(f"seed {seed}: " + "  ".join(f"cfg{i} {res[i][seed][0]:.4f} {res[i][seed][1]:.3f}s" for i in res), flush=True)
for i in res:
    accs = [a for a, _ in res[i].values()]
    times = [t for _, t in res[i].values()]
    print(f"cfg{i} {json.dumps(configs[i])}: mean acc {st.mean(accs):.5f}, mean time {st.mean(times):.3f}s, median time {st.median(times):.3f}s")
    if i:
        d = [100 * (res[i][s][0] - res[0][s][0]) for s in res[0]]
        print(f"  paired vs cfg0: {st.mean(d):+.3f} pp (sd {st.stdev(d):.3f}, SE {st.stdev(d) / len(d) ** 0.5:.3f}); "
              f"time {100 * (st.mean(times) / st.mean(t for _, t in res[0].values()) - 1):+.2f}% "
              f"(median {100 * (st.median(times) / st.median(t for _, t in res[0].values()) - 1):+.2f}%)")
