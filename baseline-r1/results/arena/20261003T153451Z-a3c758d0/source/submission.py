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
    epochs=30,
    batch_size=512,
    lr=0.4,
    momentum=0.9,
    weight_decay=5e-4,
    label_smoothing=0.1,
    warmup=0.25,
    width=64,
)


def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


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
            conv_bn(w, 2 * w),
            nn.MaxPool2d(2),
            Residual(2 * w),
            conv_bn(2 * w, 4 * w),
            nn.MaxPool2d(2),
            conv_bn(4 * w, 8 * w),
            nn.MaxPool2d(2),
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
    model = Net(context.num_classes, cfg["width"]).to(device).to(memory_format=torch.channels_last)
    state = SimpleNamespace(model=model, context=context, cfg=cfg, device=device)
    state.step_fn = torch.compile(model.features) if device.type == "cuda" else model.features
    # Warm up compilation on synthetic data (fwd + bwd); state is reset in prepare.
    if device.type == "cuda":
        bs = cfg["batch_size"]
        x = torch.randn(bs, 3, 32, 32, device=device, dtype=torch.bfloat16)
        x = x.contiguous(memory_format=torch.channels_last)
        y = torch.randint(0, context.num_classes, (bs,), device=device)
        model.train()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(state.step_fn(x), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    model = state.model
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            m.reset_parameters()
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    cfg = state.cfg
    device = state.device
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    state.optimizer = torch.optim.SGD(
        [
            dict(params=decay, weight_decay=cfg["weight_decay"]),
            dict(params=no_decay, weight_decay=0.0),
        ],
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        nesterov=True,
    )
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


def train(state) -> nn.Module:
    cfg = state.cfg
    model, opt = state.model, state.optimizer
    n = state.labels.numel()
    bs = cfg["batch_size"]
    steps_per_epoch = n // bs
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg["warmup"] * total)
    peak = cfg["lr"]
    ls = cfg["label_smoothing"]
    step = 0
    for epoch in range(cfg["epochs"]):
        x_all = augment(state.images, state.gen)
        perm = torch.randperm(n, device=state.device, generator=state.gen)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            x, y = x_all[idx], state.labels[idx]
            lr = peak * step / warm if step < warm else peak * (total - step) / (total - warm)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(state.step_fn(x), y, label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
    model.eval()
    return model
