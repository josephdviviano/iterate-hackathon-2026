p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''class Muon(torch.optim.Optimizer):
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
''','''def muon_update(params, grads, bufs, lr, momentum: float, ns_steps: int):
    for p, g, buf in zip(params, grads, bufs):
        buf.mul_(momentum).add_(g)
        g = g.add(buf, alpha=momentum)
        p.mul_(len(p) ** 0.5 / p.norm())
        update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), ns_steps)
        p.sub_(update.view(g.shape).to(p.dtype) * lr)


class Muon(torch.optim.Optimizer):
    """Muon (Keller Jordan, airbench94_muon): orthogonalized Nesterov momentum on normalized filters."""

    def __init__(self, params, lr, momentum, ns_steps, update_fn=muon_update):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        self.update_fn = update_fn
        for group in self.param_groups:
            group["lr_tensor"] = torch.zeros((), device=group["params"][0].device)
            for p in group["params"]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]
            group["lr_tensor"].fill_(group["lr"])  # a tensor, so the compiled update never recompiles
            self.update_fn(
                params,
                [p.grad for p in params],
                [self.state[p]["momentum_buffer"] for p in params],
                group["lr_tensor"],
                group["momentum"],
                group["ns_steps"],
            )
''')
rep('''        muon = Muon(filters, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"])''','''        muon = Muon(
            filters, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"], state.muon_update
        )''')
rep('''    train_net = torch.compile(net, mode=hyp["compile"]) if cuda and hyp["compile"] else net''','''    train_net = torch.compile(net, mode=hyp["compile"]) if cuda and hyp["compile"] else net
    compiled_muon = cuda and hyp["compile"]
    update = torch.compile(muon_update, mode="max-autotune-no-cudagraphs") if compiled_muon else None''')
open(p,'w').write(s)
s=open(p).read()
a='''        ema=[t.clone() for t in float_state],
    )
'''
assert s.count(a)==1
s=s.replace(a,'''        ema=[t.clone() for t in float_state],
        muon_update=update or muon_update,
    )
''')
open(p,'w').write(s)
