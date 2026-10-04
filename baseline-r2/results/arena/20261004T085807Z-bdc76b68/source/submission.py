"""Airbench-style CIFAR-100 speedrun recipe.

Whitened-patch stem + three conv groups, bf16 autocast, channels_last, GPU-side
augmentation, Nesterov SGD with label smoothing, torch.compile warmed up in build.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

ACT = ["gelu"]
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}
MEAN = torch.tensor([0.5071, 0.4865, 0.4409])
STD = torch.tensor([0.2673, 0.2564, 0.2762])

DEFAULTS = dict(
    epochs=11.25,
    batch_size=768,
    lr=14.0,  # per 1024 examples (summed loss)
    momentum=0.85,
    weight_decay=0.0153,
    bias_scaler=16.0,
    label_smoothing=0.2,
    widths=(64, 256, 768),
    whiten_kernel=2,
    depth=3,
    act="silu",
    bn_momentum=0.6,
    scaling_factor=0.2,
    translate=2,
    whiten_bias_epochs=3,
    compile_mode="max-autotune",  # CUDA graphs (the step is launch-bound) + kernel autotuning
    cudnn_benchmark=True,
    half_weights=True,
    dtype="fp16",
    alt_flip=True,
    low_res_batch_size=None,  # batch size during res_schedule stages (None: batch_size)
    lr_peak=0.23,
    lr_start=0.2,
    lr_end=0.0,
    res_schedule=((18, 4), (24, 2)),  # (resolution, epochs) stages before full 32px training
)


def make_act():
    return {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "hardswish": nn.Hardswish}[ACT[0]]()


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class Conv(nn.Conv2d):
    def __init__(self, cin, cout, kernel_size=3, padding="same", bias=False):
        super().__init__(cin, cout, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        # Identity (dirac) init of the first in_channels filters, vectorized.
        w = self.weight.data
        c, k = w.size(1), w.size(2)
        if c <= w.size(0):
            w[:c].zero_()
            i = torch.arange(c, device=w.device)
            w[i, i, k // 2, k // 2] = 1


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_momentum, depth=2):
        super().__init__()
        self.depth = depth
        if depth == 3:
            self.conv3 = Conv(cout, cout)
            self.norm3 = BatchNorm(cout, bn_momentum)
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_momentum)
        if depth >= 2:
            self.conv2 = Conv(cout, cout)
            self.norm2 = BatchNorm(cout, bn_momentum)
        self.activ = make_act()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        if self.depth == 1:
            return x
        if self.depth == 3:
            x0 = x
            x = self.activ(self.norm2(self.conv2(x)))
            x = self.activ(self.norm3(self.conv3(x)))
            return x + x0
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths, bn_momentum, scaling_factor, num_classes, whiten_kernel=2, depth=2):
        super().__init__()
        self.register_buffer("mean", MEAN.view(1, 3, 1, 1).clone(), persistent=False)
        self.register_buffer("std", STD.view(1, 3, 1, 1).clone(), persistent=False)
        whiten_width = 2 * 3 * whiten_kernel**2
        self.whiten = Conv(3, whiten_width, kernel_size=whiten_kernel, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten_act = make_act()
        self.layers = nn.Sequential(
            ConvGroup(whiten_width, widths[0], bn_momentum, depth[0]),
            ConvGroup(widths[0], widths[1], bn_momentum, depth[1]),
            ConvGroup(widths[1], widths[2], bn_momentum, depth[2]),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scaling_factor = scaling_factor

    def features(self, x):
        x = self.whiten_act(self.whiten(x))
        x = self.layers(x)
        x = F.max_pool2d(x, x.shape[2:]).flatten(1)
        return self.head(x) * self.scaling_factor

    def forward(self, x):
        # Evaluation entry point: float32 images in [0, 1].
        x = ((x - self.mean) / self.std).to(self.whiten.weight.dtype).contiguous(memory_format=torch.channels_last)
        return self.features(x).float()


def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()


@torch.no_grad()
def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.shape[2:])
    n, c, h, w = patches.shape
    flat = patches.view(n, -1)
    cov = (flat.T @ flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.copy_(torch.cat((scaled, -scaled)))


def batch_crop(images, crop_size, generator):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device, generator=generator)
    out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size]
    return out


def make_hyp(parameters):
    hyp = dict(DEFAULTS)
    hyp.update(parameters or {})
    hyp["widths"] = tuple(hyp["widths"])
    if isinstance(hyp["depth"], int):
        hyp["depth"] = (hyp["depth"],) * 3
    hyp["depth"] = tuple(hyp["depth"])
    hyp["res_schedule"] = tuple(tuple(x) for x in hyp["res_schedule"])
    return hyp


def build(context: BuildContext):
    hyp = make_hyp(context.parameters)
    device = context.device
    torch.backends.cudnn.benchmark = hyp["cudnn_benchmark"]
    ACT[0] = hyp["act"]
    model = Net(hyp["widths"], hyp["bn_momentum"], hyp["scaling_factor"], context.num_classes, hyp["whiten_kernel"], hyp["depth"])
    model = model.to(device).to(memory_format=torch.channels_last)
    if hyp["half_weights"] and device.type == "cuda":
        # fp16 conv/linear weights (airbench-style, no fp32 master copy); BN stays fp32.
        model.half()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.float()
        model.mean.data = model.mean.float()
        model.std.data = model.std.float()
    use_cuda = device.type == "cuda"
    step_fn = torch.compile(model.features, mode=hyp["compile_mode"], dynamic=False) if use_cuda else model.features
    state = SimpleNamespace(model=model, context=context, hyp=hyp, step_fn=step_fn, device=device)
    # Reusable pinned host buffer for fast host-to-device copies of the training images.
    state.pinned = torch.empty((50000, 3, 32, 32), dtype=torch.uint8).pin_memory() if use_cuda else None
    # Warm up compilation and lazy CUDA init by running the real prepare/train path
    # on synthetic data for a few steps. prepare() resets everything afterwards.
    if use_cuda:
        fake = SimpleNamespace(
            images=torch.randint(0, 256, (50000, 3, 32, 32), dtype=torch.uint8),
            labels=torch.randint(0, context.num_classes, (50000,)),
        )
        # Run every epoch (all resolutions/stages, CUDA graph recording) with a few steps each.
        for _ in range(2):
            prepare(state, fake, 0)
            train(state, warmup_steps=8)
        # Warm up eval-mode shapes (cudnn.benchmark autotuning) on synthetic inputs.
        model.eval()
        with torch.inference_mode():
            for b in (context.eval_batch_size, 10000 % context.eval_batch_size):
                if b:
                    model(torch.rand(b, 3, 32, 32, device=device))
        model.train()
        torch.cuda.synchronize()
        state.images = state.labels = state.optimizer = None
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    hyp = state.hyp
    device = state.device
    model = state.model
    for m in model.modules():
        if m is not model and hasattr(m, "reset_parameters"):
            m.reset_parameters()
    model.zero_grad(set_to_none=True)
    model.train()
    state.generator = torch.Generator(device=device)
    state.generator.manual_seed(seed)

    if state.pinned is not None and state.pinned.shape == data.images.shape:
        state.pinned.copy_(data.images)
        images = state.pinned.to(device, non_blocking=True)
    else:
        images = data.images.to(device, non_blocking=True)
    labels = data.labels.to(device, non_blocking=True)
    mean = MEAN.to(device).view(1, 3, 1, 1)
    std = STD.to(device).view(1, 3, 1, 1)
    images = ((images.float() / 255) - mean) / std
    init_whitening_conv(model.whiten, images[:5000])
    pad = hyp["translate"]
    if pad > 0:
        images = F.pad(images, (pad,) * 4, mode="reflect")
    state.images = images.to(DTYPE[hyp["dtype"]])
    state.labels = labels

    batch_size = hyp["batch_size"]
    momentum = hyp["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    whiten_bias = [model.whiten.bias]
    other = [p for n, p in model.named_parameters() if "norm" not in n and p.requires_grad and p is not model.whiten.bias]
    state.optimizer = torch.optim.SGD(
        [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=other, lr=lr, weight_decay=wd / lr),
        ],
        momentum=momentum,
        nesterov=True,
        fused=state.device.type == "cuda",
    )
    for g in state.optimizer.param_groups:
        g["initial_lr"] = g["lr"]


def stage_at(hyp, epoch):
    """Resolution and batch size used in a given (integer) epoch."""
    stage_end = 0
    for res, epochs in hyp["res_schedule"]:
        stage_end += epochs
        if epoch < stage_end:
            return res, hyp["low_res_batch_size"] or hyp["batch_size"]
    return 32, hyp["batch_size"]


def train(state, warmup_steps=None) -> nn.Module:
    hyp = state.hyp
    model = state.model
    opt = state.optimizer
    images, labels = state.images, state.labels
    n = len(images)
    epochs = hyp["epochs"]

    def lr_at(t):
        # Triangle over training progress t in [0, 1].
        peak, start, end = hyp["lr_peak"], hyp["lr_start"], hyp["lr_end"]
        if t < peak:
            return start + (1 - start) * t / peak
        return 1 + (end - 1) * (t - peak) / (1 - peak)

    gen = state.generator
    ls = hyp["label_smoothing"]
    dtype = DTYPE[hyp["dtype"]]
    epoch = 0
    flip_base = torch.rand(n, device=images.device, generator=gen) < 0.5
    while epoch < epochs:
        res, bs = stage_at(hyp, epoch)
        steps_per_epoch = n // bs
        epoch_images = batch_crop(images, 32, gen) if hyp["translate"] > 0 else images
        if res != 32:
            epoch_images = F.interpolate(epoch_images.float(), size=(res, res), mode="bilinear", antialias=True)
            epoch_images = epoch_images.to(dtype)
        if hyp["alt_flip"]:
            # Alternating flip (airbench): each image is flipped in every other epoch.
            flip = flip_base ^ bool(epoch % 2)
        else:
            flip = torch.rand(n, device=images.device, generator=gen) < 0.5
        epoch_images = torch.where(flip.view(-1, 1, 1, 1), epoch_images.flip(-1), epoch_images)
        epoch_images = epoch_images.contiguous(memory_format=torch.channels_last)
        perm = torch.randperm(n, device=images.device, generator=gen)
        if epoch >= hyp["whiten_bias_epochs"]:
            opt.param_groups[1]["initial_lr"] = 0.0
        for i in range(steps_per_epoch):
            t = (epoch + i / steps_per_epoch) / epochs
            if t >= 1 or (warmup_steps is not None and i >= warmup_steps):
                break
            idx = perm[i * bs : (i + 1) * bs]
            f = lr_at(t)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * f
            with torch.autocast("cuda", dtype=dtype):
                out = state.step_fn(epoch_images[idx])
                loss = F.cross_entropy(out, labels[idx], label_smoothing=ls, reduction="sum")
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        epoch += 1
    return model
