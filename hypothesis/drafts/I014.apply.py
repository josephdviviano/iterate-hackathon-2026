p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "epochs": 8.5,''','''    "epochs": 9.0,''')
rep('''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
''','''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    # Progressive resizing: [until_fraction_of_steps, size] pairs; later steps train at 32 px.
    "res_schedule": [[0.3, 26], [0.6, 28]],
''')
rep('''        for _ in range(2):
            prepare(state, synthetic, seed=0)
            state.whiten_bias_steps = 3
            _fit(state, total_steps=6)
''','''        sizes = sorted({32, *(size for _, size in hyp["res_schedule"])})
        for _ in range(2):
            for size in sizes:
                prepare(state, synthetic, seed=0)
                state.whiten_bias_steps = 3
                _fit(state, total_steps=6, size=size)
''')
rep('''def _fit(state, total_steps):''','''def _step_size(hyp, frac):
    for until, size in hyp["res_schedule"]:
        if frac < until:
            return size
    return 32


def _fit(state, total_steps, size=None):''')
rep('''            outputs = state.train_net(images[idx], step < state.whiten_bias_steps)''','''            x = images[idx]
            step_size = size or _step_size(hyp, step / total_steps)
            if step_size != 32:
                x = F.interpolate(x, size=(step_size, step_size), mode="bilinear", antialias=True)
                x = x.contiguous(memory_format=torch.channels_last)
            if step == 0 and size is None:
                print(f"first-batch hash: {x.float().sum().item():.6e}", flush=True)
            outputs = state.train_net(x, step < state.whiten_bias_steps)''')
open(p,'w').write(s)
