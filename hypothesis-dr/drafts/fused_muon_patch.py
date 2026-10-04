"""Draft: replace the per-parameter Muon loop with one compiled update over all conv filters.

Same math as Muon.step (Nesterov momentum, weight-norm projection, NS quintic, sqrt(max(1,out/in)) scale);
lr is a 0-d GPU tensor so schedule changes never recompile.
"""
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''class Muon(torch.optim.Optimizer):
    """Orthogonalized Nesterov momentum for conv filters, with per-step weight-norm projection (airbench94_muon)."""

    def __init__(self, params, lr, momentum, ns_steps):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                g2 = g.reshape(len(g), -1)
                scale = max(1.0, g2.size(0) / g2.size(1)) ** 0.5
                update = zeropower_via_newtonschulz5(g2, group["ns_steps"]).view(g.shape)
                p.add_(update, alpha=-group["lr"] * scale)
''', '''def _newtonschulz(G, steps: int, eps: float = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


@torch.compile(dynamic=False)
def _muon_update(params, grads, bufs, lr, momentum: float, ns_steps: int):
    with torch.no_grad():
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        updates = torch._foreach_add(grads, bufs, alpha=momentum)
        for p, g in zip(params, updates):
            p.mul_(len(p) ** 0.5 / p.norm())
            g2 = g.reshape(len(g), -1)
            scale = max(1.0, g2.size(0) / g2.size(1)) ** 0.5
            u = _newtonschulz(g2, ns_steps).view(g.shape)
            p.sub_(u.float() * (lr * scale))


class Muon:
    """Orthogonalized Nesterov momentum for conv filters, with per-step weight-norm projection (airbench94_muon).

    One compiled graph updates all filters; lr is a 0-d GPU tensor.
    """

    def __init__(self, params, lr, momentum, ns_steps):
        self.params = list(params)
        self.bufs = [torch.zeros_like(p) for p in self.params]
        self.lr_t = torch.tensor(float(lr), device=self.params[0].device)
        self.param_groups = [dict(lr=lr)]
        self.momentum = momentum
        self.ns_steps = ns_steps

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        self.lr_t.fill_(self.param_groups[0]["lr"])
        _muon_update(self.params, [p.grad for p in self.params], self.bufs, self.lr_t, self.momentum, self.ns_steps)
''')
open(p, "w").write(s)
