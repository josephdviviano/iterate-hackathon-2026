"""I129 gate: GPU time of the fused Muon step with vs without the weight-norm projection (E082-equivalent base)."""
import importlib.util, sys, torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "drafts/I125_base.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)

@torch.compile(dynamic=False)
def no_proj(params, grads, bufs, lr, momentum: float, ns_steps: int, shape_groups: tuple):
    with torch.no_grad():
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        updates = torch._foreach_add(grads, bufs, alpha=momentum)
        for idx in shape_groups:
            G = torch.stack([updates[i].reshape(len(updates[i]), -1) for i in idx])
            scale = max(1.0, G.size(1) / G.size(2)) ** 0.5
            U = m._newtonschulz(G, ns_steps)
            U = U * m.PE_NORM_MATCH[(min(G.shape[1:]), max(G.shape[1:]))]
            for j, i in enumerate(idx):
                p = params[i]
                p.sub_(U[j].view(p.shape).float() * (lr * scale))

net = m.Net(100, (128, 384, 576), 3, 0.6, 1/9).cuda().to(memory_format=torch.channels_last)
filters = [p for p in net.parameters() if p.ndim == 4 and p.requires_grad]
for p in filters: p.grad = torch.randn_like(p)
mu = m.Muon(filters, lr=0.16, momentum=0.8, ns_steps=3)
args = lambda: (mu.params, [p.grad for p in mu.params], mu.bufs, mu.lr_t, mu.momentum, mu.ns_steps, mu.shape_groups)
for name, f in (("with projection", m._muon_update), ("without projection", no_proj)):
    for _ in range(3): f(*args())
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20): f(*args())
        torch.cuda.synchronize()
    print(f"gate {name}: {sum(e.self_device_time_total for e in prof.key_averages()) / 20 / 1e3:.3f} ms/step GPU")
