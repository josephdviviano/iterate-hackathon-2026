"""Gate for I041: Muon step span (cuda events) vs sum of its kernel times (profiler), standalone and inside train_step."""
import importlib.util, sys, torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from benchmark.api import BuildContext
dev = torch.device("cuda")
state = m.build(BuildContext(dev, {}))
x = torch.rand(1024, 3, 32, 32, device=dev).to(memory_format=torch.channels_last); y = torch.randint(0, 100, (1024,), device=dev)
opts = m.make_optimizer(state.model, state.hyp, 100)
for _ in range(5): m.train_step(state, opts, x, y)
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
spans = []
for _ in range(20):
    torch.cuda.synchronize(); s.record(); opts[1].step(); e.record(); torch.cuda.synchronize(); spans.append(s.elapsed_time(e))
print(f"muon standalone span: {sum(spans)/len(spans):.3f} ms")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20): opts[1].step()
    torch.cuda.synchronize()
print(f"muon kernel sum: {sum(ev.self_device_time_total for ev in prof.key_averages())/20/1e3:.3f} ms")
# inside the real step: GPU busy vs wall
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    torch.cuda.synchronize(); s.record()
    for _ in range(20): m.train_step(state, opts, x, y)
    e.record(); torch.cuda.synchronize()
busy = sum(ev.self_device_time_total for ev in prof.key_averages()) / 20 / 1e3
print(f"train_step wall (events): {s.elapsed_time(e)/20:.3f} ms, GPU kernel sum: {busy:.3f} ms")
