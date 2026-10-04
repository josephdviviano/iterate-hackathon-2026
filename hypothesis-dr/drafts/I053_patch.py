"""Apply I053: head lr x2 with lr*wd constant."""
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''        dict(params=other, lr=lr, weight_decay=wd / lr),''', '''        dict(params=other, lr=lr * hyp["head_lr_mult"], weight_decay=wd / (lr * hyp["head_lr_mult"])),''')
rep('''    "compile": True,
''', '''    "compile": True,
    "head_lr_mult": 2.0,
''')
open(p, "w").write(s)
