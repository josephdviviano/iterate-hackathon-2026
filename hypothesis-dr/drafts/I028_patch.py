"""Apply I028: Muon group lr decays linearly to 0 at the final step (SGD group keeps the 0.07 floor)."""
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''def lr_factor(step, total_steps):''', '''def lr_factor(step, total_steps, end=0.07):''')
rep('''    return 1.0 * (1 - frac) + 0.07 * frac''', '''    return 1.0 * (1 - frac) + end * frac''')
rep('''            for g in optimizer[1].param_groups:
                g["lr"] = g["base_lr"] * f''', '''            for g in optimizer[1].param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, end=0.0)''')
open(p, "w").write(s)
