"""airbench96-style CIFAR-100 recipe: whitened stem, 3 residual conv groups, Nesterov SGD."""

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

HYP = {
    "epochs": 10.0,
    "batch_size": 1024,
    "lr": 9.0,
    "momentum": 0.85,
    "weight_decay": 0.012,
    "bias_scaler": 64.0,
    "label_smoothing": 0.3,
    "whiten_bias_epochs": 3,
    "widths": [128, 384, 576],
    "bn_momentum": 0.6,
    "scaling_factor": 1 / 9,
    "translate": 2,
    "whiten_images": 5000,
}

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False

    def reset_parameters(self):
        super().reset_parameters()


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, bn_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, bn_momentum)
        self.conv3 = Conv(channels_out, channels_out)
        self.norm3 = BatchNorm(channels_out, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x0 = x
        x = self.activ(self.norm2(self.conv2(x)))
        x = self.activ(self.norm3(self.conv3(x)))
        return x + x0


class Net(nn.Module):
    def __init__(self, num_classes, widths, bn_momentum, scaling_factor):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        whiten_width = 2 * 3 * 2 * 2
        self.whiten = nn.Conv2d(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.groups = nn.Sequential(
            ConvGroup(whiten_width, widths[0], bn_momentum),
            ConvGroup(widths[0], widths[1], bn_momentum),
            ConvGroup(widths[1], widths[2], bn_momentum),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Inputs are float [0, 1] RGB; normalization lives in the model.
        x = ((x - self.mean) / self.std).to(self.whiten.weight.dtype)
        x = x.contiguous(memory_format=torch.channels_last)
        x = F.gelu(self.whiten(x))
        x = self.groups(x)
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return (self.head(x) * self.scaling_factor).float()


def whitening_weights(images, eps=5e-4):
    """images: normalized float [N, 3, H, W]. Returns [24, 3, 2, 2] whitening filters."""
    c = images.shape[1]
    patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, c, 2, 2).float()
    flat = patches.view(patches.shape[0], -1)
    cov = (flat.T @ flat) / flat.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * 4, c, 2, 2).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    return torch.cat((scaled, -scaled))


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            m.reset_parameters()
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


def make_model(context, hyp):
    model = Net(context.num_classes, hyp["widths"], hyp["bn_momentum"], hyp["scaling_factor"])
    model = model.to(context.device).to(memory_format=torch.channels_last)
    dtype = torch.float16 if context.device.type == "cuda" else torch.float32
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.to(dtype)
    return model


def make_optimizer(model, hyp, batch_size):
    kilostep_scale = 1024 * (1 + 1 / (1 - hyp["momentum"]))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    whiten_bias = [model.whiten.bias]
    other = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and "norm" not in n and not n.startswith("whiten.")
    ]
    groups = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    opt = torch.optim.SGD(groups, momentum=hyp["momentum"], nesterov=True)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


def augment(padded, flip_mask, epoch, translate, generator=None):
    """Random crop (from reflect-padded images) plus derandomized alternating flip."""
    n, c, hp, wp = padded.shape
    h, w = hp - 2 * translate, wp - 2 * translate
    out = torch.empty((n, c, h, w), dtype=padded.dtype, device=padded.device)
    span = 2 * translate + 1
    shifts = torch.randint(0, span * span, (n,), device=padded.device)
    for s in range(span * span):
        dy, dx = divmod(s, span)
        mask = shifts == s
        out[mask] = padded[mask, :, dy : dy + h, dx : dx + w]
    flip = flip_mask if epoch % 2 == 0 else ~flip_mask
    out = torch.where(flip.view(-1, 1, 1, 1), out.flip(3), out)
    return out


def train_step(model, optimizer, x, y, label_smoothing):
    out = model(x)
    loss = F.cross_entropy(out, y, label_smoothing=label_smoothing, reduction="none").sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def build(context: BuildContext):
    hyp = {**HYP, **context.parameters}
    torch.backends.cudnn.benchmark = True
    model = make_model(context, hyp)
    state = SimpleNamespace(model=model, context=context, hyp=hyp)
    # Warm up kernels and cuDNN autotuning on synthetic data; everything is reset in prepare.
    device = context.device
    bs = hyp["batch_size"]
    if device.type == "cuda":
        opt = make_optimizer(model, hyp, bs)
        x = torch.rand(bs, 3, 32, 32, device=device, dtype=torch.float16)
        y = torch.randint(0, context.num_classes, (bs,), device=device)
        model.train()
        for _ in range(3):
            train_step(model, opt, x, y, hyp["label_smoothing"])
        model.eval()
        with torch.inference_mode():
            for b in (1, 784, 1024):
                model(torch.rand(b, 3, 32, 32, device=device))
        torch.cuda.synchronize()
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    hyp = state.hyp
    device = state.context.device
    model = state.model
    reset_model(model)
    model.train()
    dtype = model.whiten.weight.dtype
    images = data.images.to(device, non_blocking=False)
    images = images.to(dtype).div_(255)
    labels = data.labels.to(device)
    mean = torch.tensor(CIFAR_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    sub = (images[: hyp["whiten_images"]].float() - mean.float()) / std.float()
    model.whiten.weight.data.copy_(whitening_weights(sub).to(dtype))
    t = hyp["translate"]
    state.padded = F.pad(images, (t, t, t, t), mode="reflect") if t > 0 else images
    state.labels = labels
    state.n = images.shape[0]
    state.batch_size = min(hyp["batch_size"], state.n)
    state.optimizer = make_optimizer(model, hyp, state.batch_size)
    state.flip_mask = torch.rand(state.n, device=device) < 0.5


def train(state) -> nn.Module:
    hyp = state.hyp
    model, opt = state.model, state.optimizer
    n, bs = state.n, state.batch_size
    steps_per_epoch = n // bs
    total_steps = math.ceil(hyp["epochs"] * steps_per_epoch)
    schedule = np.interp(
        np.arange(1 + total_steps), [0, int(0.23 * total_steps), total_steps], [0.2, 1.0, 0.07]
    )
    whiten_bias_steps = hyp["whiten_bias_epochs"] * steps_per_epoch
    step = 0
    for epoch in range(math.ceil(hyp["epochs"])):
        epoch_images = augment(state.padded, state.flip_mask, epoch, hyp["translate"])
        perm = torch.randperm(n, device=state.padded.device)
        epoch_images = epoch_images[perm].contiguous(memory_format=torch.channels_last)
        epoch_labels = state.labels[perm]
        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * schedule[step]
            if step >= whiten_bias_steps:
                opt.param_groups[1]["lr"] = 0.0
            x = epoch_images[i * bs : (i + 1) * bs]
            y = epoch_labels[i * bs : (i + 1) * bs]
            train_step(model, opt, x, y, hyp["label_smoothing"])
            step += 1
    state.padded = None
    return model
