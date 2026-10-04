"""Apply I034: Muon group lr multiplier held at 1.0x through warmup (decay shape and floor unchanged after)."""
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''def lr_factor(step, total_steps):
    warmup = int(total_steps * 0.23)
    if step < warmup:''', '''def lr_factor(step, total_steps, warm=True):
    warmup = int(total_steps * 0.23)
    if step < warmup:
        if not warm:
            return 1.0''')
rep('''            for g in optimizer[1].param_groups:
                g["lr"] = g["base_lr"] * f''', '''            for g in optimizer[1].param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warm=False)''')
open(p, "w").write(s)
