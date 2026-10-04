p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "muon_final_lr": 0.05,  # same, for the Muon groups
''','''    "muon_final_lr": 0.05,  # same, for the Muon groups
    "muon_lr_32px": 1.25,  # Muon LR multiplier during the 32 px phase
''')
rep('''                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                if isinstance(opt, Muon):''','''                if isinstance(opt, Muon) and step_size == 32:
                    scale *= hyp["muon_lr_32px"]
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                if isinstance(opt, Muon):''')
open(p,'w').write(s)
