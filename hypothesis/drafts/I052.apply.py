p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "muon_ns_steps": 3,
''','''    "muon_ns_steps": 3,
    "muon_head": True,  # also train the linear head with Muon (without renormalization)
''')
rep('''                group["momentum"],
                group["ns_steps"],
                renorm,
            )''','''                group["momentum"],
                group["ns_steps"],
                renorm and group.get("renorm", True),
            )''')
rep('''        filters = [p for p in others if p.ndim == 4]
        others = [p for p in others if p.ndim != 4]
        muon = Muon(
            filters, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"], state.muon_update
        )''','''        filters = [p for p in others if p.ndim == 4]
        muon_groups = [dict(params=filters)]
        if hyp["muon_head"]:
            muon_groups.append(dict(params=[net.head.weight], renorm=False))
        muon_params = {id(p) for g in muon_groups for p in g["params"]}
        others = [p for p in others if id(p) not in muon_params]
        muon = Muon(
            muon_groups, hyp["muon_lr"], hyp["muon_momentum"], hyp["muon_ns_steps"], state.muon_update
        )''')
open(p,'w').write(s)
