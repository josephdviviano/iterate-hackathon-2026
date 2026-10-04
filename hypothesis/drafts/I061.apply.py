p='submissions/arena/submission.py'
s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep('''        w = self.weight.data
        nn.init.dirac_(w[: w.size(1)])''','''        # Vectorized nn.init.dirac_ (which loops in Python with a host sync per channel).
        w = self.weight.data[: self.weight.size(1)]
        w.zero_()
        w[:, :, 1, 1].diagonal().fill_(1)''')
rep('''    raw = data.images.to(device, non_blocking=True).float().div_(255)
    mean = raw.mean(dim=(0, 2, 3), keepdim=True)
    std = raw.std(dim=(0, 2, 3), keepdim=True)
    state.classifier.mean.copy_(mean)
    state.classifier.std.copy_(std)
    images = ((raw - mean) / std).to(state.dtype, memory_format=torch.channels_last)
    del raw''','''    raw = data.images.to(device, non_blocking=True).to(state.dtype)  # 0..255, exact in fp16
    var, mean = torch.var_mean(raw, dim=(0, 2, 3), keepdim=True)
    mean, std = mean.float(), var.float().sqrt()
    state.classifier.mean.copy_(mean / 255)
    state.classifier.std.copy_(std / 255)
    images = ((raw - mean.to(state.dtype)) / std.to(state.dtype)).contiguous(
        memory_format=torch.channels_last
    )
    del raw''')
open(p,'w').write(s)
