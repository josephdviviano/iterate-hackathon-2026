"""airbench96-style CIFAR-100 recipe: whitened stem, 3 residual conv groups, Nesterov SGD."""

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

HYP = {
    "epochs": 9.25,
    "batch_size": 2000,
    "lr": 9.0,
    "momentum": 0.85,
    "weight_decay": 0.006,
    "bias_scaler": 64.0,
    "label_smoothing": 0.3,
    # The whitening bias trains for this fraction of steps; afterwards the whitening output is
    # detached (no bias grad, no input-grad through group 1's first conv).
    "whiten_bias_frac": 0.2,
    "widths": [128, 384, 512],
    "bn_momentum": 0.6,
    "scaling_factor": 1 / 9,
    "translate": 2,
    "whiten_images": 5000,
    "compile": True,
    "ema_every": 5,
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    # Progressive resizing: (start fraction of training, resolution); last entry wins.
    "resolutions": [[0.0, 20], [0.3, 24], [0.4, 28], [0.7, 32]],
    # FreezeOut-style: the stem (whitening bias + group 1) LR decays to 0 at this fraction of
    # steps, after which its backward and updates are skipped. 0 disables.
    "freeze_stem_at": 0.75,
    # After the final lookahead copy, BN running stats are recomputed from this many
    # 32 px training batches (cumulative average). 0 disables.
    "bn_recal_batches": 3,
}

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.bn_momentum = momentum
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

    def forward(self, x, freeze_stem=False, freeze_whiten=False):
        # Inputs are float [0, 1] RGB; normalization lives in the model.
        x = ((x - self.mean) / self.std).to(self.whiten.weight.dtype)
        x = x.contiguous(memory_format=torch.channels_last)
        grad = torch.is_grad_enabled()
        with torch.set_grad_enabled(grad and not (freeze_stem or freeze_whiten)):
            x = F.gelu(self.whiten(x))
        with torch.set_grad_enabled(grad and not freeze_stem):
            x = self.groups[0](x)
        x = self.groups[2](self.groups[1](x))
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


def newton_schulz(G, steps=3, eps=1e-7):
    """Approximately orthogonalize G (quintic Newton-Schulz, as in Muon)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


@torch.no_grad()
def muon_update(params, grads, bufs, lr, momentum: float):
    for p, g, buf in zip(params, grads, bufs):
        g = g.float()
        buf.mul_(momentum).add_(g)
        g = g.add(buf, alpha=momentum)
        p.mul_(p.shape[0] ** 0.5 / p.float().norm())
        update = newton_schulz(g.reshape(len(g), -1)).view(g.shape)
        p.sub_(lr * update.float())


class Muon(torch.optim.Optimizer):
    """Muon on conv filters: Nesterov momentum, orthogonalized update, fixed-norm weights."""

    def __init__(self, params, lr, momentum, compiled=True):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        group = self.param_groups[0]
        self.bufs = [torch.zeros_like(p, dtype=torch.float32) for p in group["params"]]
        self.lr_t = torch.tensor(lr, device=group["params"][0].device)
        self.update = torch.compile(muon_update, mode="reduce-overhead") if compiled else muon_update

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        if group["params"][0].grad is None:  # frozen
            return
        self.lr_t.fill_(group["lr"])
        grads = [p.grad for p in group["params"]]
        self.update(group["params"], grads, self.bufs, self.lr_t, group["momentum"])


class Optimizers:
    """SGD (biases, head) and Muon (conv filters) stepped together."""

    def __init__(self, opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def make_optimizer(model, hyp, batch_size):
    kilostep_scale = 1024 * (1 + 1 / (1 - hyp["momentum"]))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    stem = lambda n: n.startswith("groups.0.")  # noqa: E731
    norm_biases = [p for n, p in named if "norm" in n and not stem(n)]
    stem_norm_biases = [p for n, p in named if "norm" in n and stem(n)]
    whiten_bias = [model.whiten.bias]
    filters = [p for n, p in named if p.ndim == 4 and not stem(n)]
    stem_filters = [p for n, p in named if p.ndim == 4 and stem(n)]
    other = [p for n, p in named if "norm" not in n and not n.startswith("whiten.") and p.ndim != 4]
    groups = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=stem_norm_biases, lr=lr_biases, weight_decay=wd / lr_biases, stem=True),
        dict(params=whiten_bias, lr=lr_biases, weight_decay=wd / lr_biases, whiten=True, stem=True),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    cuda = next(model.parameters()).is_cuda
    sgd = torch.optim.SGD(groups, momentum=hyp["momentum"], nesterov=True, fused=cuda)
    muon = Muon(filters, lr=hyp["muon_lr"], momentum=hyp["muon_momentum"], compiled=cuda)
    stem_muon = Muon(stem_filters, lr=hyp["muon_lr"], momentum=hyp["muon_momentum"], compiled=cuda)
    stem_muon.param_groups[0]["stem"] = True
    opt = Optimizers([sgd, muon, stem_muon])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


def augment(padded, flip_mask, epoch, translate):
    """Random crop (from reflect-padded images) plus derandomized alternating flip, sync-free."""
    n, c, hp, wp = padded.shape
    h, w = hp - 2 * translate, wp - 2 * translate
    device = padded.device
    dy = torch.randint(0, 2 * translate + 1, (n, 1), device=device)
    dx = torch.randint(0, 2 * translate + 1, (n, 1), device=device)
    rows = (dy + torch.arange(h, device=device)).view(n, 1, h, 1).expand(n, c, h, wp)
    out = padded.gather(2, rows)
    flip = flip_mask if epoch % 2 == 0 else ~flip_mask
    # A flipped crop reads columns right to left.
    cols = torch.arange(w, device=device).expand(n, w)
    cols = torch.where(flip.view(n, 1), w - 1 - cols, cols) + dx
    return out.gather(3, cols.view(n, 1, 1, w).expand(n, c, h, w))


class Lookahead:
    """airbench-style lookahead: every k steps, pull the net toward an EMA copy and reset it there."""

    def __init__(self, model):
        tensors = [*model.parameters(), *model.buffers()]
        self.live = [t.data for t in tensors if t.is_floating_point()]
        self.ema = [t.clone() for t in self.live]

    def update(self, decay):
        torch._foreach_lerp_(self.ema, self.live, 1 - decay)
        torch._foreach_copy_(self.live, self.ema)


@torch.no_grad()
def recalibrate_bn(model, compiled, images, batches, batch_size):
    """Recompute BN running stats for the final weights from a few train-mode forwards."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    for i in range(batches):
        compiled(images[i * batch_size : (i + 1) * batch_size])
    for m in bns:
        m.momentum = 1 - m.bn_momentum


def train_step(model, optimizer, x, y, label_smoothing, freeze_stem=False, freeze_whiten=False):
    out = model(x, freeze_stem=freeze_stem, freeze_whiten=freeze_whiten)
    loss = F.cross_entropy(out, y, label_smoothing=label_smoothing, reduction="none").sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def build(context: BuildContext):
    hyp = {**HYP, **context.parameters}
    torch.backends.cudnn.benchmark = True
    # One graph per (resolution, freeze variant); more than Dynamo's default of 8.
    torch._dynamo.config.cache_size_limit = 64
    model = make_model(context, hyp)
    # Training runs through the compiled module; evaluation uses the eager one (same parameters).
    compiled = model
    if context.device.type == "cuda" and hyp["compile"]:
        compiled = torch.compile(model, mode="max-autotune", dynamic=False)
    state = SimpleNamespace(model=model, compiled=compiled, context=context, hyp=hyp)
    # Warm up kernels and cuDNN autotuning on synthetic data; everything is reset in prepare.
    device = context.device
    bs = hyp["batch_size"]
    if device.type == "cuda":
        opt = make_optimizer(model, hyp, bs)
        # Same layout as training batches (channels_last slices); compile guards on strides.
        y = torch.randint(0, context.num_classes, (bs,), device=device)
        model.train()
        for _, res in hyp["resolutions"]:
            x = torch.rand(bs, 3, res, res, device=device, dtype=torch.float16)
            x = x.contiguous(memory_format=torch.channels_last)
            for freeze_whiten in (False, True):
                for _ in range(3):
                    train_step(compiled, opt, x, y, hyp["label_smoothing"], False, freeze_whiten)
        if hyp["freeze_stem_at"]:
            for _ in range(3):
                train_step(compiled, opt, x, y, hyp["label_smoothing"], True, True)
        if hyp["bn_recal_batches"]:
            xs = torch.rand(bs, 3, 32, 32, device=device, dtype=torch.float16)
            for _ in range(2):
                recalibrate_bn(model, compiled, xs.contiguous(memory_format=torch.channels_last), 1, bs)
        # One short synthetic trial warms prepare's and train's first-call costs (transfer,
        # eigh, padding, gathers, interpolation, lookahead). prepare resets it all.
        fake = TrainingData(
            torch.randint(0, 256, (50000, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (50000,)),
        )
        prepare(state, fake, 0)
        state.hyp = {**hyp, "epochs": 1.0}
        train(state)
        state.hyp = hyp
        model.eval()
        with torch.inference_mode():
            for b in (1, 784, 1024):
                model(torch.rand(b, 3, 32, 32, device=device))
        torch.cuda.synchronize()
        # Every training shape is compiled now; a recompile inside a trial is a bug.
        torch._dynamo.config.error_on_recompile = True
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
    model, opt = state.compiled, state.optimizer
    n, bs = state.n, state.batch_size
    steps_per_epoch = n // bs
    total_steps = math.ceil(hyp["epochs"] * steps_per_epoch)
    schedule = np.interp(
        np.arange(1 + total_steps), [0, int(0.23 * total_steps), total_steps], [0.2, 1.0, 0.07]
    )
    whiten_bias_steps = int(hyp["whiten_bias_frac"] * total_steps)
    # Stem schedule: the same warmup to the peak, then linear to 0 at the freeze point.
    peak = int(0.23 * total_steps)
    freeze_step = int(hyp["freeze_stem_at"] * total_steps) if hyp["freeze_stem_at"] else total_steps + 1
    stem_schedule = np.interp(
        np.arange(1 + total_steps), [0, peak, freeze_step, total_steps + 1], [0.2, 1.0, 0.0, 0.0]
    )
    ema_every = hyp["ema_every"]
    alpha = 0.95**5 * (np.arange(total_steps + 1) / total_steps) ** 3
    lookahead = Lookahead(state.model) if ema_every else None
    # Resolution switches at fixed fractions of the total step count.
    resolution = [
        [r for start, r in hyp["resolutions"] if k >= start * total_steps][-1]
        for k in range(total_steps)
    ]
    step = 0
    for epoch in range(math.ceil(hyp["epochs"])):
        epoch_images = augment(state.padded, state.flip_mask, epoch, hyp["translate"])
        perm = torch.randperm(n, device=state.padded.device)
        epoch_steps = resolution[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]
        versions = {}
        for res in sorted(set(epoch_steps)):
            images = epoch_images
            if res != images.shape[-1]:
                images = F.interpolate(
                    images, size=(res, res), mode="bilinear", antialias=True, align_corners=False
                )
            versions[res] = images[perm].contiguous(memory_format=torch.channels_last)
        epoch_labels = state.labels[perm]
        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * (stem_schedule if g.get("stem") else schedule)[step]
            if step >= whiten_bias_steps:
                for g in opt.param_groups:
                    if g.get("whiten"):
                        g["lr"] = 0.0
            x = versions[resolution[step]][i * bs : (i + 1) * bs]
            y = epoch_labels[i * bs : (i + 1) * bs]
            train_step(
                model, opt, x, y, hyp["label_smoothing"], step >= freeze_step, step >= whiten_bias_steps
            )
            step += 1
            if lookahead is not None and step % ema_every == 0:
                lookahead.update(float(alpha[step]))
    if lookahead is not None:
        lookahead.update(1.0)
    if hyp["bn_recal_batches"]:
        k = hyp["bn_recal_batches"] * bs
        images = augment(state.padded, state.flip_mask, epoch + 1, hyp["translate"])
        images = images[torch.randperm(n, device=images.device)[:k]]
        recalibrate_bn(
            state.model, model, images.contiguous(memory_format=torch.channels_last),
            hyp["bn_recal_batches"], bs,
        )
    state.padded = None
    return state.model
