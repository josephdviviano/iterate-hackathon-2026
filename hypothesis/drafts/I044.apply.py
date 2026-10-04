p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''def muon_update(shape_groups, grads, bufs, lr, momentum: float, ns_steps: int):''','''def muon_update(shape_groups, grads, bufs, lr, momentum: float, ns_steps: int, renorm: bool):''')
rep('''        for p, u in zip(params, U):
            p.mul_(len(p) ** 0.5 / p.norm())
            p.sub_(u.view(p.shape).to(p.dtype) * lr)''','''        for p, u in zip(params, U):
            if renorm:
                p.mul_(len(p) ** 0.5 / p.norm())
            p.sub_(u.view(p.shape).to(p.dtype) * lr)''')
rep('''    @torch.no_grad()
    def step(self):
        for group in self.param_groups:''','''    total_steps = 1  # set by prepare; filters are renormalized every 2 + int(15 * progress) steps
    steps_done = 0
    next_renorm = 0

    @torch.no_grad()
    def step(self):
        renorm = self.steps_done >= self.next_renorm
        if renorm:
            self.next_renorm = self.steps_done + 2 + int(15 * self.steps_done / self.total_steps)
        self.steps_done += 1
        for group in self.param_groups:''')
rep('''                group["momentum"],
                group["ns_steps"],
            )''','''                group["momentum"],
                group["ns_steps"],
                renorm,
            )''')
rep('''    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)''','''    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)
    for opt in state.optimizers:
        if isinstance(opt, Muon):
            opt.total_steps = state.total_steps''')
open(p,'w').write(s)
