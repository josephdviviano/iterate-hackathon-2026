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
    epochs=18,
    batch_size=512,
    lr=0.4,
    momentum=0.9,
    weight_decay=1e-3,
    label_smoothing=0.2,
    warmup=0.25,
    width=64,
)


def conv_bn(c_in, c_out, pool=False):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = conv_bn(c, c)
        self.b = conv_bn(c, c)

    def forward(self, x):
        return x + self.b(self.a(x))


class Net(nn.Module):
    def __init__(self, num_classes, w):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.body = nn.Sequential(
            conv_bn(3, w),
            conv_bn(w, 2 * w, pool=True),
            Residual(2 * w),
            conv_bn(2 * w, 4 * w, pool=True),
            conv_bn(4 * w, 8 * w, pool=True),
            Residual(8 * w),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(8 * w, num_classes, bias=False)
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
    model = Net(context.num_classes, cfg["width"]).to(device).to(memory_format=torch.channels_last)
    state = SimpleNamespace(model=model, context=context, cfg=cfg, device=device)
    features = torch.compile(model.features, dynamic=False) if cuda else model.features
    bs = cfg["batch_size"]
    state.x = torch.zeros(bs, 3, 32, 32, device=device, dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    state.y = torch.zeros(bs, dtype=torch.long, device=device)
    state.lr = torch.zeros((), device=device)
    params = list(model.parameters())
    decay = [p for p in params if p.ndim > 1]
    state.bufs = [torch.zeros_like(p) for p in params]
    mom, wd, ls = cfg["momentum"], cfg["weight_decay"], cfg["label_smoothing"]

    def step():
        # Forward, backward and nesterov SGD (coupled weight decay) on static buffers.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = F.cross_entropy(features(state.x), state.y, label_smoothing=ls)
        loss.backward()
        with torch.no_grad():
            grads = [p.grad for p in params]
            torch._foreach_add_([p.grad for p in decay], decay, alpha=wd)
            torch._foreach_mul_(state.bufs, mom)
            torch._foreach_add_(state.bufs, grads)
            torch._foreach_add_(grads, state.bufs, alpha=mom)
            torch._foreach_mul_(grads, state.lr)
            torch._foreach_sub_(params, grads)

    model.train()
    if cuda:
        # Warm up on a side stream, then capture the whole step as one CUDA graph.
        state.x.normal_()
        state.y.random_(0, context.num_classes)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                step()
        torch.cuda.current_stream().wait_stream(side)
        model.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        state.step = graph.replay
        # Warm up the remaining kernels and the allocator with a short synthetic trial.
        n = 50_000
        fake = TrainingData(
            torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (n,)),
        )
        prepare(state, fake, 0)
        train(state, max_steps=10)
        torch.cuda.synchronize()
    else:

        def eager_step():
            model.zero_grad(set_to_none=True)
            step()

        state.step = eager_step
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


def augment(padded, gen):
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
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return x.contiguous(memory_format=torch.channels_last)


def train(state, max_steps=None) -> nn.Module:
    cfg = state.cfg
    model = state.model
    n = state.labels.numel()
    bs = cfg["batch_size"]
    steps_per_epoch = n // bs
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg["warmup"] * total)
    peak = cfg["lr"]
    step = 0
    for epoch in range(cfg["epochs"]):
        x_all = augment(state.images, state.gen)
        perm = torch.randperm(n, device=state.device, generator=state.gen)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            torch.index_select(x_all, 0, idx, out=state.x)
            torch.index_select(state.labels, 0, idx, out=state.y)
            lr = peak * step / warm if step < warm else peak * (total - step) / (total - warm)
            state.lr.fill_(lr)
            state.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                return model
        # Keep the CPU from running far ahead of the GPU (slows the first trial).
        if state.device.type == "cuda":
            torch.cuda.synchronize()
    model.eval()
    return model
