"""Apply I012 (Muon on 4D conv filters except the frozen whitening conv) to submissions/arena/submission.py."""
p = "submissions/arena/submission.py"
s = open(p).read()

s = s.replace('''    "compile": True,
}''', '''    "compile": True,
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    "ns_steps": 5,
}''')

muon = '''
@torch.compile(dynamic=False)
def zeropower_via_newtonschulz5(G, steps: int, eps: float = 1e-7):
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


class Muon(torch.optim.Optimizer):
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


'''
s = s.replace("class Lookahead:", muon + "class Lookahead:")

s = s.replace('''    other = [
        p
        for k, p in model.named_parameters()
        if "norm" not in k and p.requires_grad and p is not model.whiten.bias
    ]
    groups = [
        dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    optimizer = torch.optim.SGD(groups, momentum=momentum, nesterov=True)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]
    return optimizer''', '''    filters = [p for k, p in model.named_parameters() if p.ndim == 4 and p.requires_grad]
    other = [
        p
        for k, p in model.named_parameters()
        if "norm" not in k and p.requires_grad and p is not model.whiten.bias and p.ndim != 4
    ]
    groups = [
        dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    sgd = torch.optim.SGD(groups, momentum=momentum, nesterov=True)
    muon = Muon(filters, lr=hyp["muon_lr"], momentum=hyp["muon_momentum"], ns_steps=hyp["ns_steps"])
    for opt in (sgd, muon):
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return [sgd, muon]''')

s = s.replace('''    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()''', '''    for opt in optimizer:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizer:
        opt.step()''')

s = s.replace('''            for gi, g in enumerate(optimizer.param_groups):
                g["lr"] = g["base_lr"] * f if (gi != 0 or whiten_on) else 0.0''', '''            for gi, g in enumerate(optimizer[0].param_groups):
                g["lr"] = g["base_lr"] * f if (gi != 0 or whiten_on) else 0.0
            for g in optimizer[1].param_groups:
                g["lr"] = g["base_lr"] * f''')
open(p, "w").write(s)
