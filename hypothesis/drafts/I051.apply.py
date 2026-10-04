p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "batch_size": 1536,
''','''    "batch_size": 1536,
    "lowres_batch_size": 2048,  # batch size for epochs below 32 px; 0 uses batch_size throughout
''')
rep('''            for size in sizes:
                prepare(state, synthetic, seed=0)
                state.whiten_bias_steps = 3
                _fit(state, total_steps=6, size=size)''','''            for size in sizes:
                prepare(state, synthetic, seed=0)
                state.whiten_bias_steps = 3
                _fit(state, total_steps=6, size=size, batch_size=_epoch_batch(state, size))''')
rep('''    state.batch_size = batch_size
    state.steps_per_epoch = len(data.labels) // batch_size
    state.total_steps = math.ceil(hyp["epochs"] * state.steps_per_epoch)
    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)''','''    for group in state.optimizer.param_groups:
        group["wd_per_example"] = group["weight_decay"] / batch_size

    state.batch_size = batch_size
    state.steps_per_epoch = len(data.labels) // batch_size
    # Per-epoch plan: (size, batch size, steps); low-res epochs may use a larger batch.
    state.plan = []
    for epoch in range(math.ceil(hyp["epochs"])):
        size = _epoch_size(hyp, epoch / hyp["epochs"])
        bs = _epoch_batch(state, size)
        steps = len(data.labels) // bs
        steps = math.ceil(min(1.0, hyp["epochs"] - epoch) * steps)
        state.plan.append((size, bs, steps))
    state.total_steps = sum(steps for _, _, steps in state.plan)
    state.whiten_bias_steps = sum(
        steps for epoch, (_, _, steps) in enumerate(state.plan) if epoch < hyp["whiten_bias_epochs"]
    )''')
rep('''def _fit(state, total_steps, size=None):
    hyp, net, optimizer = state.hyp, state.net, state.optimizer
    labels, batch_size, steps_per_epoch = state.labels, state.batch_size, state.steps_per_epoch
    warmup_steps = int(total_steps * hyp["warmup"])
    ema_decay = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    step = 0
    net.train()
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):''','''def _epoch_batch(state, size):
    lowres = state.hyp["lowres_batch_size"]
    return min(lowres, len(state.labels)) if lowres and size != 32 else state.batch_size


def _fit(state, total_steps, size=None, batch_size=None):
    hyp, net = state.hyp, state.net
    labels = state.labels
    if size is None:
        plan = state.plan
    else:  # warmup: one shape for a few steps
        plan = [(size, batch_size, total_steps)]
    total_samples = sum(bs * steps for _, bs, steps in plan)
    warmup = hyp["warmup"]
    seen = 0
    step = 0
    net.train()
    for epoch, (epoch_size, batch_size, epoch_steps) in enumerate(plan):
        for group in state.optimizer.param_groups:
            group["weight_decay"] = group["wd_per_example"] * batch_size''')
rep('''        epoch_size = size or _epoch_size(hyp, epoch * steps_per_epoch / total_steps)
        if epoch_size != 32:''','''        if epoch_size != 32:''')
rep('''        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = order[i * batch_size : (i + 1) * batch_size]''','''        for i in range(epoch_steps):
            idx = order[i * batch_size : (i + 1) * batch_size]''')
rep('''            if step < warmup_steps:
                frac = step / warmup_steps
                scale = 0.2 * (1 - frac) + frac
            else:
                frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                scale = (1 - frac) + hyp["final_lr"] * frac''','''            progress = seen / total_samples  # schedules follow samples seen, not steps
            if progress < warmup:
                frac = progress / warmup
                scale = 0.2 * (1 - frac) + frac
            else:
                frac = (progress - warmup) / (1 - warmup)
                scale = (1 - frac) + hyp["final_lr"] * frac''')
rep('''            step += 1
            if hyp["ema_every"] and step % hyp["ema_every"] == 0:
                _lookahead(state, ema_decay[step].item())''','''            step += 1
            seen += batch_size
            if hyp["ema_every"] and step % hyp["ema_every"] == 0:
                _lookahead(state, 0.95**5 * (seen / total_samples) ** 3)''')
open(p,'w').write(s)
