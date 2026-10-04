p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''            nn.AdaptiveMaxPool2d(1),''','''            GlobalMaxPool(),''')
rep('''class Net(nn.Module):''','''class GlobalMaxPool(nn.Module):
    """Global max over H, W; compiles to a fused reduction (AdaptiveMaxPool2d's backward uses slow atomics)."""

    def forward(self, x):
        return x.flatten(2).max(dim=2).values


class Net(nn.Module):''')
open(p,'w').write(s)
