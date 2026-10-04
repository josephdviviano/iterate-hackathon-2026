"""I077 calibration: first-step update RMS of the SGD head (x2) vs an NS3-orthogonalised head update, on real batch 0, seed 0."""
import importlib.util, sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "submissions/arena/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from benchmark.api import BuildContext
from benchmark.data import load_split
dev = torch.device("cuda")
data = load_split(Path("data"), train=True)
state = m.build(BuildContext(dev, {"compile": False}))
torch.manual_seed(0); m.prepare(state, data, 0)
hyp = state.hyp
imgs = m.augment(state.padded, state.flip_bits, 0, hyp["translate"], hyp["contrast"], hyp["brightness"], False)
perm = torch.randperm(len(imgs), device=dev)
x = m.downsample(imgs[perm][:hyp["batch_size"]].contiguous(memory_format=torch.channels_last), hyp["res_schedule"][0][1])
y = state.labels[perm][:hyp["batch_size"]]
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = F.cross_entropy(state.model(x).float(), y, label_smoothing=hyp["label_smoothing"], reduction="sum")
loss.backward()
head = state.model.head.weight
g = head.grad
sgd = state.optimizer[0].param_groups[2]
lr, wd, mom = sgd["base_lr"], sgd["weight_decay"], hyp["momentum"]
d = g + wd * head.detach()
sgd_update = lr * (1 + mom) * d  # Nesterov first step (buf = d)
mu_mom = hyp["muon_momentum"]
u = (1 + mu_mom) * g
ns = m._newtonschulz(u[None], hyp["ns_steps"])[0].float()
rms = lambda t: t.pow(2).mean().sqrt().item()
print(f"head lr {lr:.6f} wd {wd:.6f} lr*wd {lr*wd:.6e}")
print(f"SGD first-step update RMS: {rms(sgd_update):.6e}; NS(u) RMS: {rms(ns):.6e}")
print(f"matched head Muon lr = {rms(sgd_update)/rms(ns):.6f}")
