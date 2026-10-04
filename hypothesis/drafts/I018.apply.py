import subprocess
p='submissions/arena/submission.py'
s=subprocess.check_output(['git','show','04e09da:'+p],text=True)
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''            if step_size != 32:
                x = F.interpolate(x, size=(step_size, step_size), mode="bilinear", antialias=True)
                x = x.contiguous(memory_format=torch.channels_last)''','''            if step_size != 32:
                x = batch_random_crop(x, step_size)''')
rep('''def batch_cutout(images, size):''','''def batch_random_crop(images, size):
    """Independent random size x size crop per image, without rescaling."""
    n, c, h, w = images.shape
    dev = images.device
    oy = torch.randint(0, h - size + 1, (n, 1), device=dev)
    ox = torch.randint(0, w - size + 1, (n, 1), device=dev)
    span = torch.arange(size, device=dev)
    rows = (oy + span)[:, None, :, None]
    cols = (ox + span)[:, None, None, :]
    batch = torch.arange(n, device=dev)[:, None, None, None]
    chans = torch.arange(c, device=dev)[None, :, None, None]
    return images[batch, chans, rows, cols].contiguous(memory_format=torch.channels_last)


def batch_cutout(images, size):''')
open(p,'w').write(s)
