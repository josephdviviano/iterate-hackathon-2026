p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
''','''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    "muon_lr": 0.24,  # Muon for the 3x3 conv filters; 0 keeps them on SGD
    "muon_momentum": 0.6,
    "muon_ns_steps": 3,
''')
rep('''#############################################
#               Augmentation                #''','''#############################################
#                   Muon                    #
#############################################


def zeropower_via_newtonschulz5(G, steps, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon (Keller Jordan, airbench94_muon): orthogonalized Nesterov momentum on normalized filters."""

    def __init__(self, params, lr, momentum, ns_steps):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                buf = self.state[p].get("momentum_buffer")
                if buf is None:
                    buf = self.state[p]["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), group["ns_steps"])
                p.add_(update.view(g.shape).to(p.dtype), alpha=-group["lr"])


#############################################
#               Augmentation                #''')
rep('''    others = [p for name, p in net.named_parameters() if "norm" not in name and p.requires_grad]
    state.optimizer = torch.optim.SGD(''','''    others = [p for name, p in net.named_parameters() if "norm" not in name and p.requires_grad]
    state.optimizers = []
    if hyp["muon_lr"]:
        filters = [p for p in others if p.ndim == 4]
        others = [p for p in others if p.ndim != 4]
        muon = Muon(filters, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"])
        state.optimizers.append(muon)
    state.optimizer = torch.optim.SGD(''')
rep('''    for group in state.optimizer.param_groups:
        group["initial_lr"] = group["lr"]
''','''    state.optimizers.append(state.optimizer)
    for opt in state.optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
''')
rep('''            optimizer.zero_grad(set_to_none=True)
            loss.backward()''','''            for opt in state.optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()''')
rep('''            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            optimizer.step()''','''            for opt in state.optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                opt.step()''')
open(p,'w').write(s)
