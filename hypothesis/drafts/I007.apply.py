p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
''','''    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    # Progressive resizing: [until_fraction_of_steps, size] pairs; later epochs train at 32 px.
    "res_schedule": [[0.4, 24], [0.7, 28]],
''')
rep('''            nn.MaxPool2d(3),
''','''            nn.AdaptiveMaxPool2d(1),
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
rep('''def _fit(state, total_steps):''','''def _epoch_size(hyp, frac):
    for until, size in hyp["res_schedule"]:
        if frac < until:
            return size
    return 32


def _fit(state, total_steps, size=None):''')
rep('''        if epoch % 2 == 1:
            images = images.flip(-1)
''','''        if epoch % 2 == 1:
            images = images.flip(-1)
        epoch_size = size or _epoch_size(hyp, epoch * steps_per_epoch / total_steps)
        if epoch_size != 32:
            images = F.interpolate(
                images, size=(epoch_size, epoch_size), mode="bilinear", antialias=True
            ).contiguous(memory_format=torch.channels_last)
''')
open(p,'w').write(s)
