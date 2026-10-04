p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
''','''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    "select_from_epoch": 3,  # selective backprop: from this epoch, train on the highest-loss examples
    "select_keep": 614,  # examples kept per batch (fixed size keeps compiled shapes static)
''')
rep('''            state.whiten_bias_steps = 3
            _fit(state, total_steps=6)
''','''            state.whiten_bias_steps = 3
            _fit(state, total_steps=6)
            prepare(state, synthetic, seed=0)
            state.whiten_bias_steps = 3
            _fit(state, total_steps=6, select_from_step=0)
''')
rep('''def _fit(state, total_steps):
    hyp, net, optimizer = state.hyp, state.net, state.optimizer
''','''def _fit(state, total_steps, select_from_step=None):
    hyp, net, optimizer = state.hyp, state.net, state.optimizer
    if select_from_step is None:
        select_from_step = math.ceil(hyp["select_from_epoch"] * state.steps_per_epoch)
    bn_stats = [
        t for m in net.modules() if isinstance(m, nn.BatchNorm2d)
        for t in (m.running_mean, m.running_var)
    ]
''')
rep('''            idx = order[i * batch_size : (i + 1) * batch_size]
            outputs = state.train_net(images[idx], step < state.whiten_bias_steps)
            loss = F.cross_entropy(
                outputs.float(),
                labels[idx],
''','''            idx = order[i * batch_size : (i + 1) * batch_size]
            x, y = images[idx], labels[idx]
            if step >= select_from_step and hyp["select_keep"] < len(idx):
                with torch.no_grad():
                    saved = [t.clone() for t in bn_stats]
                    scores = F.cross_entropy(state.train_net(x, False).float(), y, reduction="none")
                    torch._foreach_copy_(bn_stats, saved)
                    keep = scores.topk(hyp["select_keep"], sorted=False).indices
                x, y = x[keep], y[keep]
            outputs = state.train_net(x, step < state.whiten_bias_steps)
            loss = F.cross_entropy(
                outputs.float(),
                y,
''')
open(p,'w').write(s)
