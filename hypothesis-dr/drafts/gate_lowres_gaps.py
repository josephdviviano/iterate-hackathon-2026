"""I101 gate: per-step wall (cuda events) vs summed kernel time for the 20/28/32 px train steps on the tip."""
import importlib.util, sys, torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
dev = torch.device("cuda")
state = m.build(BuildContext(dev, {}))
bs = state.hyp["batch_size"]
x32 = torch.rand(bs, 3, 32, 32, device=dev).to(memory_format=torch.channels_last); y = torch.randint(0, 100, (bs,), device=dev)
opts = m.make_optimizer(state.model, state.hyp, 100)
state.model.whiten_bias_grad = False
for size, steps in ((20, 48), (28, 64), (32, 104)):
    x = m.downsample(x32, size) if size < 32 else x32
    for _ in range(5): m.train_step(state, opts, x, y)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): m.train_step(state, opts, x, y)
    e.record(); torch.cuda.synchronize(); wall = s.elapsed_time(e) / 20
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20): m.train_step(state, opts, x, y)
        torch.cuda.synchronize()
    busy = sum(ev.self_device_time_total for ev in prof.key_averages()) / 20 / 1e3
    print(f"gate {size}px: wall {wall:.2f} ms, kernels {busy:.2f} ms, gap {wall-busy:.2f} ms/step x {steps} steps = {(wall-busy)*steps/1e3:.3f} s/trial")
