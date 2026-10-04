"""Apply I035: slice the 31x31 whitening output to 30x30 so pooling never discards a computed row/column."""
p = "submissions/arena/submission.py"
s = open(p).read()
a = '''        x = F.silu(self.whiten(x))'''
assert a in s
s = s.replace(a, '''        x = F.silu(self.whiten(x)[..., :30, :30])''')
open(p, "w").write(s)
