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
    # [resolution, epochs, frozen leading blocks] in training order. Frozen blocks run
    # without gradients (FreezeOut-style), which makes the full-resolution stage cheap.
    stages=[[16, 4, 0], [24, 1, 1], [24, 2, 2], [32, 1, 2], [32, 3, 3]],
    batch_size=768,
    lr=0.5,
    momentum=0.9,
    weight_decay=1e-3,
    label_smoothing=0.2,
    pad=2,  # random-crop translation range in pixels
    warmup=0.25,
    widths=[32, 128, 320, 640],
    act="relu",
    logit_scale=0.125,
    bn_weight=True,  # train BatchNorm weights (else frozen at 1)
    muon=True,  # orthogonalized (Newton-Schulz) momentum updates for conv filters
    muon_lr=0.14,
    muon_momentum=0.6,
    ns_steps=3,
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


def muon_update(p, g, buf, lr, momentum, ns_steps=3):
    buf.mul_(momentum).add_(g)
    g = g.add(buf, alpha=momentum)
    p.mul_(len(p) ** 0.5 / p.norm())  # fixed filter norm instead of weight decay
    p.sub_(newton_schulz(g.reshape(len(g), -1), ns_steps).view(p.shape) * lr)


def conv_bn(c_in, c_out, act, pool=False):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers += [nn.BatchNorm2d(c_out), ACTS[act]()]
    return nn.Sequential(*layers)


class GlobalMaxPool(nn.Module):
    def forward(self, x):
        # max with indices: amax's tie-splitting backward goes NaN under inductor here.
        return x.flatten(2).max(dim=2).values


class Residual(nn.Module):
    def __init__(self, c, act):
        super().__init__()
        self.a = conv_bn(c, c, act)
        self.b = conv_bn(c, c, act)

    def forward(self, x):
        return x + self.b(self.a(x))


class Net(nn.Module):
    def __init__(self, num_classes, widths, act="relu", scale=0.125):
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
        self.scale = scale

    def segments(self, frozen=0):
        # Compiled separately so the head's gradients arrive before the stem's.
        # The first `frozen` blocks form their own (no-grad) segment.
        stem, tail = self.body[frozen:-3], self.body[-3:]
        segs = [stem, lambda x: self.fc(tail(x)) * self.scale]
        return [self.body[:frozen], *segs] if frozen else segs

    def features(self, x):
        for f in self.segments():
            x = f(x)
        return x

    def forward(self, x):
        # Evaluation input: float32 in [0, 1]. Training feeds pre-normalized bf16.
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.features(x).float()


def build(context: BuildContext):
    torch.backends.cudnn.benchmark = True
    cfg = {**DEFAULTS, **(context.parameters or {})}
    device = context.device
    cuda = device.type == "cuda"
    model = Net(context.num_classes, cfg["widths"], cfg["act"], cfg["logit_scale"]).to(device).to(memory_format=torch.channels_last)
    state = SimpleNamespace(model=model, context=context, cfg=cfg, device=device)
    bs = cfg["batch_size"]
    state.y = torch.zeros(bs, dtype=torch.long, device=device)
    state.lr = torch.zeros((), device=device)
    if not cfg["bn_weight"]:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    conv = [i for i, p in enumerate(params) if p.ndim == 4] if cfg["muon"] else []
    sgd = [i for i in range(len(params)) if i not in conv]
    decay = [i for i in sgd if params[i].ndim > 1]
    state.bufs = [torch.zeros_like(p) for p in params]
    mom, wd, ls = cfg["momentum"], cfg["weight_decay"], cfg["label_smoothing"]
    muon_scale, muon_mom = cfg["muon_lr"] / cfg["lr"], cfg["muon_momentum"]
    ns = cfg["ns_steps"]
    muon_fn = torch.compile(muon_update, dynamic=False) if cuda else muon_update
    side = torch.cuda.Stream() if cuda else None

    def make_step(frozen):
        # Forward, backward and nesterov SGD (coupled weight decay) on static buffers.
        # Muon updates run on a side stream as soon as each filter gradient is ready,
        # overlapping with the rest of the backward pass. The first `frozen` blocks
        # run without gradients and are not updated.
        segs = model.segments(frozen)
        if cuda:
            segs = [torch.compile(f, dynamic=False) for f in segs]
        fixed = {id(p) for p in model.body[:frozen].parameters()}
        active = [i for i, p in enumerate(params) if id(p) not in fixed]
        act_conv = [i for i in conv if i in active]
        act_sgd = [i for i in sgd if i in active]
        act_decay = [i for i in decay if i in active]

        def step(x):
            if frozen:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                    x = segs[0](x)
            for f in segs[1 if frozen else 0:]:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    x = f(x)
            update(x, active, act_conv, act_sgd, act_decay)

        return step

    def update(logits, active, conv, sgd, decay):
        muon_lr = state.lr * muon_scale
        main = torch.cuda.current_stream() if cuda else None

        def hook(i):
            def run(g):
                if cuda:
                    side.wait_stream(main)
                    with torch.cuda.stream(side), torch.no_grad():
                        muon_fn(params[i], g, state.bufs[i], muon_lr, muon_mom, ns)
                else:
                    with torch.no_grad():
                        muon_fn(params[i], g, state.bufs[i], muon_lr, muon_mom, ns)

            return run

        handles = [params[i].register_hook(hook(i)) for i in conv]
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = F.cross_entropy(logits, state.y, label_smoothing=ls)
        grads = [None] * len(params)
        for i, g in zip(active, torch.autograd.grad(loss, [params[i] for i in active])):
            grads[i] = g
        for h in handles:
            h.remove()
        if cuda and conv:
            main.wait_stream(side)
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

    model.train()
    state.xs, state.steps = {}, {}
    for res, frozen in sorted({(r, f) for r, _, f in cfg["stages"]}):
        step = make_step(frozen)
        if res not in state.xs:
            x = torch.zeros(bs, 3, res, res, device=device, dtype=torch.bfloat16)
            state.xs[res] = x.contiguous(memory_format=torch.channels_last)
        x = state.xs[res]
        if cuda and cfg.get("graph", True):
            # Warm up on a side stream, then capture the whole step as one CUDA graph.
            x.normal_()
            state.y.random_(0, context.num_classes)
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    step(x)
            torch.cuda.current_stream().wait_stream(warm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step(x)
            state.steps[res, frozen] = graph.replay
        else:
            state.steps[res, frozen] = lambda x=x, step=step: step(x)
    if cuda and cfg.get("graph", True):
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
    pad = state.cfg["pad"]
    state.images = F.pad(images, (pad,) * 4, mode="reflect")
    state.labels = data.labels.to(device, non_blocking=True)
    state.gen = torch.Generator(device=device).manual_seed(seed)


def augment(padded, gen, flip=None, h=32, w=32):
    n, c, hp, wp = padded.shape
    dev = padded.device
    dy = torch.randint(0, hp - h + 1, (n,), device=dev, generator=gen)
    dx = torch.randint(0, wp - w + 1, (n,), device=dev, generator=gen)
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
    epochs = [(r, f) for r, e, f in cfg["stages"] for _ in range(e)]
    total = len(epochs) * steps_per_epoch
    warm = int(cfg["warmup"] * total)
    peak = cfg["lr"]
    step = 0
    flip = None
    if cfg["alt_flip"]:
        flip = torch.rand(n, device=state.device, generator=state.gen) < 0.5
    for res, frozen in epochs:
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
            state.steps[res, frozen]()
            step += 1
            if max_steps is not None and step >= max_steps:
                return model
        # Keep the CPU from running far ahead of the GPU (slows the first trial).
        if state.device.type == "cuda":
            torch.cuda.synchronize()
    model.eval()
    return model
