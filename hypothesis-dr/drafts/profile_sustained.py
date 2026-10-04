"""Diagnostic: tight-loop step time over a sustained run vs a short burst, and per-epoch train() time."""
import importlib.util, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.api import BuildContext
from benchmark.data import load_split
sub = Path(sys.argv[1] if len(sys.argv) > 1 else "submissions/arena") / "submission.py"
spec = importlib.util.spec_from_file_location("sub", sub); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_num_threads(4)
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = mod.build(BuildContext(dev, {}))
torch.manual_seed(0)
mod.prepare(state, data, 0)
imgs = mod.augment(state.padded, state.flip_bits, 0, 2).contiguous(memory_format=torch.channels_last)
opt = state.optimizer
for block in range(8):
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(48):
        mod.train_step(state, opt, imgs[i*1024:(i+1)*1024], state.labels[i*1024:(i+1)*1024])
    torch.cuda.synchronize(); print(f"block {block}: {(time.perf_counter()-t)/48*1e3:.2f} ms/step")
time.sleep(3)
torch.manual_seed(0); mod.prepare(state, data, 0); torch.cuda.synchronize()
t = time.perf_counter(); mod.train(state); torch.cuda.synchronize(); print(f"train(): {time.perf_counter()-t:.3f}s for {state.total_steps} steps")
