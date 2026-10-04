"""airbench-style CIFAR-100 speedrun recipe.

Frozen patch-whitening first conv, Dirac-initialized conv groups, BatchNorm with a
scaled bias learning rate, Nesterov SGD with Lookahead and a triangular schedule,
label smoothing, alternating flip and 2 px translate, all data on the GPU in fp16.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

HYP = {
    "epochs": 7.5,
    "batch_size": 1024,
    "lr": 9.0,
    "momentum": 0.85,
    "weight_decay": 0.012,
    "bias_scaler": 64.0,
    "label_smoothing": 0.3,
    "whiten_bias_epochs": 3,
    "translate": 2,
    "widths": (128, 384, 768),
    "convs_per_group": 3,
    "bn_momentum": 0.6,
    "scaling_factor": 1 / 9,
    "compile": True,
}

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


# ----------------------------------------------------------------------------- model


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, n_convs, bn_momentum):
        super().__init__()
        self.convs = nn.ModuleList([Conv(cin if i == 0 else cout, cout) for i in range(n_convs)])
        self.norms = nn.ModuleList([BatchNorm(cout, bn_momentum) for _ in range(n_convs)])
        self.pool = nn.MaxPool2d(2)
        self.activ = nn.GELU()

    def forward(self, x):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            if i == 0:
                x = self.pool(x)
            x = self.activ(norm(x))
        return x


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        n, m = hyp["convs_per_group"], hyp["bn_momentum"]
        whiten_width = 2 * 3 * 2 * 2
        self.whiten = nn.Conv2d(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, w1, n, m),
            ConvGroup(w1, w2, n, m),
            ConvGroup(w2, w3, n, m),
            nn.MaxPool2d(3),
            nn.Flatten(),
        )
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scale = hyp["scaling_factor"]

    def forward(self, x):
        # x: normalized fp16 channels_last images
        return self.head(self.layers(self.whiten(x))) * self.scale


class Classifier(nn.Module):
    """Evaluation entry point: float32 [0, 1] images in, float32 logits out."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = ((x - self.mean) / self.std).half().contiguous(memory_format=torch.channels_last)
        return self.net(x).float()


# ----------------------------------------------------------------------------- whitening


def init_whitening_conv(layer, images, eps=5e-4):
    # images: [N, 3, 32, 32] normalized
    h, w = layer.weight.shape[2:]
    c = images.shape[1]
    patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c * h * w).float()
    cov = (patches.T @ patches) / len(patches)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.dtype)


# ----------------------------------------------------------------------------- data


def augment_epoch(padded, flip_bits, epoch, translate):
    """Random 32x32 crops from reflect-padded images plus alternating flip. No host sync."""
    n = padded.shape[0]
    dev = padded.device
    size = padded.shape[-1] - 2 * translate
    shifts = torch.randint(0, 2 * translate + 1, (n, 2), device=dev)
    ar = torch.arange(size, device=dev)
    rows = (shifts[:, 0:1] + ar).view(n, 1, size, 1)
    cols = (shifts[:, 1:2] + ar).view(n, 1, 1, size)
    flip = (flip_bits ^ (epoch % 2)).bool().view(n, 1, 1, 1)
    cols = torch.where(flip, padded.shape[-1] - 1 - cols, cols)
    idx_n = torch.arange(n, device=dev).view(n, 1, 1, 1)
    idx_c = torch.arange(3, device=dev).view(1, 3, 1, 1)
    out = padded[idx_n, idx_c, rows, cols]
    return out.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- api


def build(context: BuildContext):
    hyp = {**HYP, **context.parameters}
    device = context.device
    net = Net(hyp, context.num_classes).to(device).half().to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    model = Classifier(net).to(device)
    train_net = net
    if hyp["compile"] and device.type == "cuda":
        train_net = torch.compile(net)
    state = SimpleNamespace(hyp=hyp, context=context, model=model, net=net, train_net=train_net)

    # Warm up compilation on synthetic data (untimed); prepare() resets everything after.
    if device.type == "cuda":
        bs = hyp["batch_size"]
        x = torch.randn(bs, 3, 32, 32, device=device).half().contiguous(
            memory_format=torch.channels_last
        )
        y = torch.randint(0, context.num_classes, (bs,), device=device)
        net.train()
        for _ in range(3):
            out = train_net(x)
            loss = F.cross_entropy(out, y, label_smoothing=hyp["label_smoothing"], reduction="none").sum()
            loss.backward()
            for p in net.parameters():
                p.grad = None
        torch.cuda.synchronize()
    return state


def reset_model(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.reset_parameters()
        elif isinstance(mod, nn.BatchNorm2d):
            mod.reset_parameters()  # also resets running stats
    for p in net.parameters():
        p.grad = None


def prepare(state, data: TrainingData, seed: int) -> None:
    hyp, device = state.hyp, state.context.device
    net = state.net
    reset_model(net)
    net.train()

    images = data.images.to(device, non_blocking=True)
    labels = data.labels.to(device, non_blocking=True)
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    images = ((images.float() / 255 - mean) / std).half()
    init_whitening_conv(net.whiten, images[:5000])
    t = hyp["translate"]
    state.padded = F.pad(images, (t, t, t, t), mode="reflect") if t else images
    state.labels = labels
    state.flip_bits = torch.randint(0, 2, (len(images),), device=device, dtype=torch.uint8)

    bs, momentum = hyp["batch_size"], hyp["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * bs / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    whiten_bias = [net.whiten.bias]
    norm_biases = [p for k, p in net.named_parameters() if "norms" in k and p.requires_grad]
    other = [
        p
        for k, p in net.named_parameters()
        if "norms" not in k and p.requires_grad and p is not net.whiten.bias
    ]
    groups = [
        dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    state.optimizer = torch.optim.SGD(groups, momentum=momentum, nesterov=True)
    for g in state.optimizer.param_groups:
        g["base_lr"] = g["lr"]


def lr_factor(step, total):
    warmup = int(total * 0.23)
    if step < warmup:
        frac = step / warmup
        return 0.2 * (1 - frac) + 1.0 * frac
    frac = (step - warmup) / (total - warmup)
    return 1.0 * (1 - frac) + 0.07 * frac


def train(state) -> nn.Module:
    hyp = state.hyp
    net, train_net, opt = state.net, state.train_net, state.optimizer
    bs = hyp["batch_size"]
    n = len(state.labels)
    steps_per_epoch = n // bs
    total = math.ceil(steps_per_epoch * hyp["epochs"])
    alpha = [0.95**5 * (s / total) ** 3 for s in range(total + 1)]
    ema = [t.detach().clone() for t in net.state_dict().values()]
    live = [t for t in net.state_dict().values()]
    ema_pairs = [(e, l) for e, l in zip(ema, live) if l.dtype in (torch.half, torch.float)]
    ls = hyp["label_smoothing"]

    step = 0
    epoch = 0
    while step < total:
        inputs_all = augment_epoch(state.padded, state.flip_bits, epoch, hyp["translate"])
        perm = torch.randperm(n, device=inputs_all.device)
        if epoch >= hyp["whiten_bias_epochs"]:
            opt.param_groups[0]["base_lr"] = 0.0
        for b in range(steps_per_epoch):
            if step >= total:
                break
            idx = perm[b * bs : (b + 1) * bs]
            out = train_net(inputs_all[idx])
            loss = F.cross_entropy(out, state.labels[idx], label_smoothing=ls, reduction="none").sum()
            loss.backward()
            f = lr_factor(step, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                decay = alpha[step]
                with torch.no_grad():
                    for e, l in ema_pairs:
                        e.lerp_(l, 1 - decay)
                        l.copy_(e)
        epoch += 1
    with torch.no_grad():
        for e, l in ema_pairs:
            l.copy_(e)
    state.padded = None
    return state.model
