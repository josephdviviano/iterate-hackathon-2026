"""ResNet9-style CIFAR-100 speedrun recipe: GPU-resident data, bf16, torch.compile."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

DEFAULTS = dict(
    stages=[[32, 10]],  # [resolution, epochs] in training order
    batch_size=768,
    lr=0.5,
    momentum=0.9,
    weight_decay=1e-3,
    label_smoothing=0.2,
    warmup=0.25,
    widths=[32, 128, 256, 512],
    act="relu",
    muon=True,  # orthogonalized (Newton-Schulz) momentum updates for conv filters
    muon_lr=0.14,
    muon_momentum=0.6,
    alt_flip=True,
)

ACTS = dict(relu=lambda: nn.ReLU(inplace=True), gelu=nn.GELU, silu=lambda: nn.SiLU(inplace=True))


def newton_schulz(g, steps=3, eps=1e-7):
    # Approximately orthogonalize g (Muon); quintic iteration coefficients from Keller Jordan.
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    tall = g.size(0) > g.size(1)
    if tall:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return x.T if tall else x


def muon_update(p, g, buf, lr, momentum):
    buf.mul_(momentum).add_(g)
    g = g.add(buf, alpha=momentum)
    p.mul_(len(p) ** 0.5 / p.norm())  # fixed filter norm instead of weight decay
    p.sub_(newton_schulz(g.reshape(len(g), -1)).view(p.shape) * lr)


def conv_bn(c_in, c_out, act, pool=False):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers += [nn.BatchNorm2d(c_out), ACTS[act]()]
    return nn.Sequential(*layers)


class GlobalMaxPool(nn.Module):
    def forward(self, x):
        return x.amax(dim=(2, 3))


class Residual(nn.Module):
    def __init__(self, c, act):
        super().__init__()
        self.a = conv_bn(c, c, act)
        self.b = conv_bn(c, c, act)

    def forward(self, x):
        return x + self.b(self.a(x))


class Net(nn.Module):
    def __init__(self, num_classes, widths, act="relu"):
        super().__init__()
        w = widths
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.body = nn.Sequential(
            conv_bn(3, w[0], act),
            conv_bn(w[0], w[1], act, pool=True),
            Residual(w[1], act),
            conv_bn(w[1], w[2], act, pool=True),
            conv_bn(w[2], w[3], act, pool=True),
            Residual(w[3], act),
            GlobalMaxPool(),
        )
        self.fc = nn.Linear(w[3], num_classes, bias=False)
        self.scale = 0.125

    def features(self, x):
        return self.fc(self.body(x)) * self.scale

    def forward(self, x):
        # Evaluation input: float32 in [0, 1]. Training feeds pre-normalized bf16.
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.features(x).float()


def build(context: BuildContext):
    cfg = {**DEFAULTS, **(context.parameters or {})}
    device = context.device
    cuda = device.type == "cuda"
    model = Net(context.num_classes, cfg["widths"], cfg["act"]).to(device).to(memory_format=torch.channels_last)
    state = SimpleNamespace(model=model, context=context, cfg=cfg, device=device)
    features = torch.compile(model.features, dynamic=False) if cuda else model.features
    bs = cfg["batch_size"]
    state.y = torch.zeros(bs, dtype=torch.long, device=device)
    state.lr = torch.zeros((), device=device)
    params = list(model.parameters())
    conv = [i for i, p in enumerate(params) if p.ndim == 4] if cfg["muon"] else []
    sgd = [i for i in range(len(params)) if i not in conv]
    decay = [i for i in sgd if params[i].ndim > 1]
    state.bufs = [torch.zeros_like(p) for p in params]
    mom, wd, ls = cfg["momentum"], cfg["weight_decay"], cfg["label_smoothing"]
    muon_scale, muon_mom = cfg["muon_lr"] / cfg["lr"], cfg["muon_momentum"]
    muon_fn = torch.compile(muon_update, dynamic=False) if cuda else muon_update

    def step(x):
        # Forward, backward and nesterov SGD (coupled weight decay) on static buffers.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = F.cross_entropy(features(x), state.y, label_smoothing=ls)
        grads = list(torch.autograd.grad(loss, params))
        with torch.no_grad():
            torch._foreach_add_([grads[i] for i in decay], [params[i] for i in decay], alpha=wd)
            p_sgd = [params[i] for i in sgd]
            g_sgd = [grads[i] for i in sgd]
            b_sgd = [state.bufs[i] for i in sgd]
            torch._foreach_mul_(b_sgd, mom)
            torch._foreach_add_(b_sgd, g_sgd)
            torch._foreach_add_(g_sgd, b_sgd, alpha=mom)
            torch._foreach_mul_(g_sgd, state.lr)
            torch._foreach_sub_(p_sgd, g_sgd)
            for i in conv:
                muon_fn(params[i], grads[i], state.bufs[i], state.lr * muon_scale, muon_mom)

    model.train()
    state.xs, state.steps = {}, {}
    for res in sorted({r for r, _ in cfg["stages"]}):
        x = torch.zeros(bs, 3, res, res, device=device, dtype=torch.bfloat16)
        x = x.contiguous(memory_format=torch.channels_last)
        state.xs[res] = x
        if cuda:
            # Warm up on a side stream, then capture the whole step as one CUDA graph.
            x.normal_()
            state.y.random_(0, context.num_classes)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step(x)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step(x)
            state.steps[res] = graph.replay
        else:
            state.steps[res] = lambda x=x: step(x)
    if cuda:
        # Warm up the remaining kernels and the allocator with a short synthetic trial.
        n = 50_000
        fake = TrainingData(
            torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (n,)),
        )
        prepare(state, fake, 0)
        train(state, max_steps=10)
        torch.cuda.synchronize()
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    # Reset everything learned, in place (the CUDA graph holds these addresses).
    model = state.model
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        for b in state.bufs:
            b.zero_()
    model.train()
    device = state.device
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    images = data.images.to(device, non_blocking=True).float().div_(255)
    images = ((images - mean) / std).to(torch.bfloat16)
    state.images = F.pad(images, (4, 4, 4, 4), mode="reflect")
    state.labels = data.labels.to(device, non_blocking=True)
    state.gen = torch.Generator(device=device).manual_seed(seed)


def augment(padded, gen, flip=None):
    n, c, hp, wp = padded.shape
    h, w = hp - 8, wp - 8
    dev = padded.device
    dy = torch.randint(0, 9, (n,), device=dev, generator=gen)
    dx = torch.randint(0, 9, (n,), device=dev, generator=gen)
    ar_h = torch.arange(h, device=dev)
    ar_w = torch.arange(w, device=dev)
    rows = (dy[:, None] + ar_h)[:, None, :, None].expand(n, c, h, wp)
    x = padded.gather(2, rows)
    cols = (dx[:, None] + ar_w)[:, None, None, :].expand(n, c, h, w)
    x = x.gather(3, cols)
    if flip is None:
        flip = torch.rand(n, device=dev, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return x.contiguous(memory_format=torch.channels_last)


def train(state, max_steps=None) -> nn.Module:
    cfg = state.cfg
    model = state.model
    n = state.labels.numel()
    bs = cfg["batch_size"]
    steps_per_epoch = n // bs
    epochs = [r for r, e in cfg["stages"] for _ in range(e)]
    total = len(epochs) * steps_per_epoch
    warm = int(cfg["warmup"] * total)
    peak = cfg["lr"]
    step = 0
    flip = None
    if cfg["alt_flip"]:
        flip = torch.rand(n, device=state.device, generator=state.gen) < 0.5
    for res in epochs:
        x_all = augment(state.images, state.gen, flip)
        if flip is not None:
            flip = ~flip
        if res != 32:
            x_all = F.interpolate(x_all.float(), size=(res, res), mode="bilinear", antialias=True)
            x_all = x_all.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        x_static = state.xs[res]
        perm = torch.randperm(n, device=state.device, generator=state.gen)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            torch.index_select(x_all, 0, idx, out=x_static)
            torch.index_select(state.labels, 0, idx, out=state.y)
            lr = peak * step / warm if step < warm else peak * (total - step) / (total - warm)
            state.lr.fill_(lr)
            state.steps[res]()
            step += 1
            if max_steps is not None and step >= max_steps:
                return model
        # Keep the CPU from running far ahead of the GPU (slows the first trial).
        if state.device.type == "cuda":
            torch.cuda.synchronize()
    model.eval()
    return model
