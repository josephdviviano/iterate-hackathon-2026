"""I134 gate: can the block-3 NS group (576x5184 pair, PE-NS3) overlap the backward of the network on a side stream?"""
import importlib.util, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("sub", "drafts/I131_base.py"); m = importlib.util.module_from_spec(spec); sys.modules["sub"] = m; spec.loader.exec_module(m)
from benchmark.api import BuildContext
dev = torch.device("cuda")
state = m.build(BuildContext(dev, {}))
model = state.model; model.whiten_bias_grad = False
x = torch.rand(1536, 3, 32, 32, device=dev).to(memory_format=torch.channels_last); y = torch.randint(0, 100, (1536,), device=dev)
G = torch.randn(2, 576, 5184, device=dev)
ns = torch.compile(lambda G: m._newtonschulz(G, 3), dynamic=False)
side = torch.cuda.Stream()
def fwd():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return F.cross_entropy(state.step_model(x).float(), y, reduction="sum")
def run(mode):
    loss = fwd()
    if mode == "serial":
        loss.backward(); ns(G)
    elif mode == "bwd":
        loss.backward()
    else:
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            ns(G)
        loss.backward()
        torch.cuda.current_stream().wait_stream(side)
    model.zero_grad(set_to_none=True)
for mode in ("serial", "overlap", "bwd", "serial", "overlap"):
    for _ in range(3): run(mode)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(30): run(mode)
    torch.cuda.synchronize(); print(f"gate {mode}: {(time.perf_counter() - t) / 30 * 1e3:.2f} ms/iter (fwd+bwd[+NS])")
