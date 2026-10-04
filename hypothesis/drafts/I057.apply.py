p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "muon_head": True,  # also train the linear head with Muon (without renormalization)
''','''    "muon_head": True,  # also train the linear head with Muon (without renormalization)
    "muon_head_lr_scale": 0.5,  # head Muon LR relative to the filters'
''')
rep('''            muon_groups.append(dict(params=[net.head.weight], renorm=False))''','''            head_lr = hyp["muon_lr"] * hyp["muon_head_lr_scale"]
            muon_groups.append(dict(params=[net.head.weight], renorm=False, lr=head_lr))''')
open(p,'w').write(s)
