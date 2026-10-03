"""Airbench-style CIFAR-100 speedrun recipe.

Whitening stem + three conv groups, bf16 autocast, channels_last, torch.compile
warmed up on synthetic data in build (untimed). Nesterov SGD with label
smoothing, lookahead EMA, GPU-side flip/translate augmentation.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

HYP = {
    "widths": (128, 384, 576),
    "depth": 3,  # convs per group; depth 3 adds a residual around conv2/conv3
    "epochs": 7.5,
    "batch_size": 1024,
    "lr": 9.0,
    "momentum": 0.85,
    "weight_decay": 0.012,
    "bias_scaler": 64.0,
    "label_smoothing": 0.2,
    "whiten_bias_epochs": 3,
    "translate": 2,
    "bn_momentum": 0.6,
    "scale": 1 / 9,
    "lookahead": True,
    "compile": True,
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    "ns_steps": 3,
}


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False

    def reset_parameters(self):
        super().reset_parameters()


class Conv(nn.Conv2d):
    def __init__(self, cin, cout, kernel_size=3, padding="same", bias=False):
        super().__init__(cin, cout, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, depth, bn_momentum):
        super().__init__()
        self.depth = depth
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_momentum)
        if depth == 3:
            self.conv3 = Conv(cout, cout)
            self.norm3 = BatchNorm(cout, bn_momentum)
        self.activ = nn.SiLU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        if self.depth == 3:
            x0 = x
            x = self.activ(self.norm2(self.conv2(x)))
            x = self.activ(self.norm3(self.conv3(x)) + x0)
        else:
            x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, num_classes, widths, depth, bn_momentum, scale):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        whiten_width = 2 * 3 * 2 * 2
        self.whiten = Conv(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.groups = nn.Sequential(
            ConvGroup(whiten_width, widths[0], depth, bn_momentum),
            ConvGroup(widths[0], widths[1], depth, bn_momentum),
            ConvGroup(widths[1], widths[2], depth, bn_momentum),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scale = scale

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = F.silu(self.whiten(x))
        x = self.groups(x)
        x = F.max_pool2d(x, x.shape[-1]).flatten(1)
        return self.head(x) * self.scale


@torch.no_grad()
def init_whitening(layer, images, eps=5e-4):
    c, (h, w) = images.shape[1], layer.weight.shape[2:]
    patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    flat = patches.view(len(patches), -1)
    cov = (flat.T @ flat) / len(flat)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.copy_(torch.cat((scaled, -scaled)))


def augment(padded, flip_bits, epoch, r):
    """Random translate by up to r pixels (from reflect-padded images) + alternating flip."""
    n = len(padded)
    device = padded.device
    shifts = torch.randint(0, 2 * r + 1, (n, 2), device=device)
    base = torch.arange(32, device=device)
    rows = (base[None, :] + shifts[:, :1])[:, None, :, None]
    cols = (base[None, :] + shifts[:, 1:])[:, None, None, :]
    idx_n = torch.arange(n, device=device)[:, None, None, None]
    idx_c = torch.arange(3, device=device)[None, :, None, None]
    out = padded[idx_n, idx_c, rows, cols]
    flip = flip_bits if epoch % 2 == 0 else ~flip_bits
    out = torch.where(flip[:, None, None, None], out.flip(-1), out)
    return out



@torch.compile(dynamic=False)
def zeropower_via_newtonschulz5(G, steps: int, eps: float = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Orthogonalized Nesterov momentum for conv filters, with per-step weight-norm projection (airbench94_muon)."""

    def __init__(self, params, lr, momentum, ns_steps):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                g2 = g.reshape(len(g), -1)
                scale = max(1.0, g2.size(0) / g2.size(1)) ** 0.5
                update = zeropower_via_newtonschulz5(g2, group["ns_steps"]).view(g.shape)
                p.add_(update, alpha=-group["lr"] * scale)


class Lookahead:
    def __init__(self, model):
        self.ema = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model, decay):
        for k, v in model.state_dict().items():
            if k in self.ema:
                self.ema[k].lerp_(v, 1 - decay)
                v.copy_(self.ema[k])


def make_optimizer(model, hyp, total_steps):
    bs = hyp["batch_size"]
    momentum = hyp["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * bs / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    whiten_bias = [model.whiten.bias]
    norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
    filters = [p for k, p in model.named_parameters() if p.ndim == 4 and p.requires_grad]
    other = [
        p
        for k, p in model.named_parameters()
        if "norm" not in k and p.requires_grad and p is not model.whiten.bias and p.ndim != 4
    ]
    groups = [
        dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    sgd = torch.optim.SGD(groups, momentum=momentum, nesterov=True)
    muon = Muon(filters, lr=hyp["muon_lr"], momentum=hyp["muon_momentum"], ns_steps=hyp["ns_steps"])
    for opt in (sgd, muon):
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return [sgd, muon]


def lr_factor(step, total_steps):
    warmup = int(total_steps * 0.23)
    if step < warmup:
        frac = step / warmup
        return 0.2 * (1 - frac) + 1.0 * frac
    frac = (step - warmup) / max(1, total_steps - warmup)
    return 1.0 * (1 - frac) + 0.07 * frac


def build(context: BuildContext):
    hyp = {**HYP, **context.parameters}
    device = context.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = Net(context.num_classes, hyp["widths"], hyp["depth"], hyp["bn_momentum"], hyp["scale"])
    model = model.to(device).to(memory_format=torch.channels_last)
    step_model = torch.compile(model, mode="max-autotune-no-cudagraphs") if hyp["compile"] and device.type == "cuda" else model
    state = SimpleNamespace(model=model, step_model=step_model, context=context, hyp=hyp)
    # Warm up compilation and cuDNN autotuning on synthetic data (state is reset in prepare).
    bs = hyp["batch_size"]
    x = torch.rand(bs, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
    y = torch.randint(0, context.num_classes, (bs,), device=device)
    model.train()
    opt = make_optimizer(model, hyp, 10)
    for _ in range(3):
        train_step(state, opt, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return state


def train_step(state, optimizer, x, y):
    with torch.autocast(state.context.device.type, dtype=torch.bfloat16):
        logits = state.step_model(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=state.hyp["label_smoothing"], reduction="sum")
    for opt in optimizer:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizer:
        opt.step()


def prepare(state, data: TrainingData, seed: int) -> None:
    hyp = state.hyp
    device = state.context.device
    model = state.model
    for m in model.modules():
        if m is not model and hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    images = data.images.to(device, non_blocking=True).float().div_(255)
    labels = data.labels.to(device, non_blocking=True)
    mean = images.mean(dim=(0, 2, 3), keepdim=True)
    std = images.std(dim=(0, 2, 3), keepdim=True)
    with torch.no_grad():
        model.mean.copy_(mean)
        model.std.copy_(std)
        init_whitening(model.whiten, ((images[:5000] - mean) / std))
    r = hyp["translate"]
    state.padded = F.pad(images, (r, r, r, r), mode="reflect") if r > 0 else images
    state.labels = labels
    state.flip_bits = torch.rand(len(images), device=device) < 0.5
    steps_per_epoch = len(images) // hyp["batch_size"]
    state.total_steps = int(steps_per_epoch * hyp["epochs"])
    state.optimizer = make_optimizer(model, hyp, state.total_steps)
    state.lookahead = Lookahead(model) if hyp["lookahead"] else None
    model.train()


def train(state) -> nn.Module:
    hyp = state.hyp
    model = state.model
    optimizer = state.optimizer
    total = state.total_steps
    bs = hyp["batch_size"]
    n = len(state.labels)
    r = hyp["translate"]
    alpha = ((0.95**5) * (torch.arange(total + 1) / total) ** 3).tolist()
    step = 0
    epoch = 0
    while step < total:
        whiten_on = epoch < hyp["whiten_bias_epochs"]
        imgs = augment(state.padded, state.flip_bits, epoch, r) if r > 0 else state.padded
        perm = torch.randperm(n, device=imgs.device)
        imgs = imgs[perm].contiguous(memory_format=torch.channels_last)
        labels = state.labels[perm]
        for i in range(n // bs):
            if step >= total:
                break
            f = lr_factor(step, total)
            for gi, g in enumerate(optimizer[0].param_groups):
                g["lr"] = g["base_lr"] * f if (gi != 0 or whiten_on) else 0.0
            for g in optimizer[1].param_groups:
                g["lr"] = g["base_lr"] * f
            train_step(state, optimizer, imgs[i * bs : (i + 1) * bs], labels[i * bs : (i + 1) * bs])
            step += 1
            if state.lookahead is not None and step % 5 == 0:
                state.lookahead.update(model, decay=alpha[step])
        epoch += 1
    if state.lookahead is not None:
        state.lookahead.update(model, decay=1.0)
    model.eval()
    return model
