"""Diagnostic: track where training goes nonfinite."""
import importlib.util, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.api import BuildContext
from benchmark.data import load_split
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_num_threads(4)
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = mod.build(BuildContext(dev, {}))
torch.manual_seed(0); mod.prepare(state, data, 0)
print("after prepare finite:", all(torch.isfinite(p).all().item() for p in state.model.parameters()))
orig = mod.train_step
cnt = [0]
def step(state, opt, x, y):
    orig(state, opt, x, y); cnt[0] += 1
    bad = [k for k, p in state.model.named_parameters() if not torch.isfinite(p).all()]
    if bad or cnt[0] <= 2:
        print(cnt[0], "nonfinite params:", bad[:5]); 
        if bad: sys.exit()
mod.train_step = step
mod.train(state)
print("done", cnt[0])
