p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''def zeropower_via_newtonschulz5(G, steps, eps=1e-7):
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
''','''def zeropower_via_newtonschulz5(G, steps, eps=1e-7):
    """Batched over the leading dim: G is [batch, rows, cols]; each matrix is normalized separately."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X


def muon_update(shape_groups, grads, bufs, lr, momentum: float, ns_steps: int):
    """One Muon step for lists of same-shape filters; the Newton-Schulz runs batched per shape."""
    for params, gs, bs in zip(shape_groups, grads, bufs):
        for g, buf in zip(gs, bs):
            buf.mul_(momentum).add_(g)
        G = torch.stack([g.add(buf, alpha=momentum).flatten(1) for g, buf in zip(gs, bs)])
        U = zeropower_via_newtonschulz5(G, ns_steps)
        for p, u in zip(params, U):
            p.mul_(len(p) ** 0.5 / p.norm())
            p.sub_(u.view(p.shape).to(p.dtype) * lr)


class Muon(torch.optim.Optimizer):
    """Muon (Keller Jordan, airbench94_muon): orthogonalized Nesterov momentum on normalized filters."""

    def __init__(self, params, lr, momentum, ns_steps, update_fn=muon_update):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        self.update_fn = update_fn
        for group in self.param_groups:
            group["lr_tensor"] = torch.zeros((), device=group["params"][0].device)
            shapes = {}
            for p in group["params"]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                shapes.setdefault(tuple(p.shape), []).append(p)
            group["shape_groups"] = list(shapes.values())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            shape_groups = group["shape_groups"]
            group["lr_tensor"].fill_(group["lr"])  # a tensor, so the compiled update never recompiles
            self.update_fn(
                shape_groups,
                [[p.grad for p in ps] for ps in shape_groups],
                [[self.state[p]["momentum_buffer"] for p in ps] for ps in shape_groups],
                group["lr_tensor"],
                group["momentum"],
                group["ns_steps"],
            )
''')
rep('''        muon = Muon(filters, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"])''','''        muon = Muon(
            filters, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"], state.muon_update
        )''')
rep('''    train_net = torch.compile(net, mode=hyp["compile"]) if cuda and hyp["compile"] else net''','''    train_net = torch.compile(net, mode=hyp["compile"]) if cuda and hyp["compile"] else net
    if cuda and hyp["compile"]:
        update = torch.compile(muon_update, mode="max-autotune-no-cudagraphs")
    else:
        update = muon_update''')
rep('''        ema=[t.clone() for t in float_state],
    )
''','''        ema=[t.clone() for t in float_state],
        muon_update=update,
    )
''')
open(p,'w').write(s)
