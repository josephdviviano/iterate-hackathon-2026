p='submissions/arena/submission.py'
s=open(p).read()
a='''        momentum=momentum,
        nesterov=True,
    )'''
assert s.count(a)==1
s=s.replace(a,'''        momentum=momentum,
        nesterov=True,
        fused=state.device.type == "cuda",
    )''')
open(p,'w').write(s)
