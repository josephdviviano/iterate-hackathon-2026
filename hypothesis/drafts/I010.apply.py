p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''    "cutout": 0,
''','''    "cutout": 0,
    "contrast": 0.15,  # per-image contrast factor drawn from [1 - c, 1 + c]
    "brightness": 0.1,  # per-image offset drawn from [-b, b], in normalized units
''')
rep('''        if hyp["cutout"]:
''','''        if hyp["contrast"] or hyp["brightness"]:
            images = batch_color_jitter(images, hyp["contrast"], hyp["brightness"])
        if hyp["cutout"]:
''')
rep('''def batch_cutout(images, size):''','''def batch_color_jitter(images, contrast, brightness):
    n = len(images)
    shape = (n, 1, 1, 1)
    c = torch.empty(shape, device=images.device).uniform_(1 - contrast, 1 + contrast)
    b = torch.empty(shape, device=images.device).uniform_(-brightness, brightness)
    mean = images.mean(dim=(1, 2, 3), keepdim=True, dtype=torch.float32)
    return ((images - mean) * c + (mean + b)).to(images.dtype)


def batch_cutout(images, size):''')
open(p,'w').write(s)
