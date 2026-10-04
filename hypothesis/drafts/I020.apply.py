import subprocess
p='submissions/arena/submission.py'
s=subprocess.check_output(['git','show','ed8dc49:'+p],text=True)
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
''','''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    "bn_reestimate_batches": 5,  # recompute BN running stats at 32 px after training; 0 disables
''')
rep('''def train(state) -> nn.Module:
    _fit(state, state.total_steps)
    return state.classifier
''','''def train(state) -> nn.Module:
    _fit(state, state.total_steps)
    _reestimate_bn(state, state.hyp["bn_reestimate_batches"])
    return state.classifier


@torch.no_grad()
def _reestimate_bn(state, batches):
    """Recompute BN running stats with the final weights on 32 px training batches."""
    if not batches:
        return
    norms = [m for m in state.net.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    images = batch_crop(state.images, 32) if state.hyp["translate"] else state.images
    idx = torch.randperm(len(images), device=images.device)[: batches * state.batch_size]
    state.net.train()
    for chunk in idx.split(state.batch_size):
        state.net(images[chunk])
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
''')
open(p,'w').write(s)
