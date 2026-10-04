"""airbench96-style CIFAR-100 speedrun recipe.

Frozen 2x2 patch-whitening conv, three residual conv groups (3 convs each), dirac init,
BatchNorm with fixed scale, GELU, Nesterov SGD with label smoothing and Lookahead.
The whole dataset lives on the GPU; augmentation (alternating flip, translate, cutout)
runs on the GPU once per epoch.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

DEFAULTS = {
    "epochs": 11,
    "batch_size": 1024,
    "lr": 9.0,  # per 1024 examples
    "momentum": 0.85,
    "weight_decay": 0.012,  # per 1024 examples, decoupled from lr
    "bias_scaler": 64.0,
    "label_smoothing": 0.1,
    "whiten_bias_epochs": 3,
    "whiten_images": 50000,
    "flip": "alternate",
    "translate": 2,
    "cutout": 0,
    "widths": [96, 256, 576],
    "bn_momentum": 0.4,
    "scaling_factor": 1 / 9,
    "compile_mode": "max-autotune",
    "warmup_steps": 3,
}

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout, kernel_size=3, padding="same", bias=False):
        super().__init__(cin, cout, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        if w.size(2) == 3:
            torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_momentum):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_momentum)
        self.conv3 = Conv(cout, cout)
        self.norm3 = BatchNorm(cout, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x0 = x
        x = self.activ(self.norm2(self.conv2(x)))
        x = self.activ(self.norm3(self.conv3(x)))
        return x0 + x


class Net(nn.Module):
    def __init__(self, widths, bn_momentum, scaling_factor, num_classes):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1), persistent=False)
        whiten_width = 2 * 3 * 2**2
        self.whiten = Conv(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0], bn_momentum),
            ConvGroup(widths[0], widths[1], bn_momentum),
            ConvGroup(widths[1], widths[2], bn_momentum),
            nn.MaxPool2d(3),
            nn.Flatten(),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            x = (x - self.mean) / self.std
            x = x.contiguous(memory_format=torch.channels_last)
            x = self.whiten(x)
            x = self.layers(x)
            x = self.head(x) * self.scaling_factor
        return x.float()


class TrainLoss(nn.Module):
    def __init__(self, net, label_smoothing):
        super().__init__()
        self.net = net
        self.label_smoothing = label_smoothing

    def forward(self, x, y):
        logits = self.net(x)
        return F.cross_entropy(logits, y, label_smoothing=self.label_smoothing, reduction="sum")


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()  # weight=1, bias=0, running stats reset


@torch.no_grad()
def init_whitening(layer, images, eps=5e-4, chunk=5000):
    """images: normalized float [N,3,32,32]. Uncentered 2x2 patch second moment, as airbench."""
    kh, kw = layer.weight.shape[2:]
    d = 3 * kh * kw
    cov = torch.zeros(d, d, device=images.device, dtype=torch.float64)
    n = 0
    for x in images.split(chunk):
        p = x.float().unfold(2, kh, 1).unfold(3, kw, 1)  # N,C,H',W',kh,kw
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, d)
        cov += (p.T @ p).double()
        n += p.shape[0]
    cov = (cov / n).float()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(d, 3, kh, kw).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data.copy_(torch.cat((scaled, -scaled)))


def augment_epoch(padded, flip_mask, epoch, cfg, generator=None):
    """padded: [N,3,32+2r,32+2r] in [0,1]; returns [N,3,32,32] augmented."""
    n = padded.shape[0]
    dev = padded.device
    r = cfg["translate"]
    if r > 0:
        shifts = torch.randint(0, 2 * r + 1, (n, 2), device=dev)
        ar = torch.arange(32, device=dev)
        iy = (shifts[:, 0:1] + ar).view(n, 1, 32, 1)
        ix = (shifts[:, 1:2] + ar).view(n, 1, 1, 32)
        idx_n = torch.arange(n, device=dev).view(n, 1, 1, 1)
        idx_c = torch.arange(3, device=dev).view(1, 3, 1, 1)
        out = padded[idx_n, idx_c, iy, ix]
    else:
        out = padded.clone()
    if cfg["flip"] == "alternate":
        mask = flip_mask if epoch % 2 == 0 else ~flip_mask
        out = torch.where(mask.view(-1, 1, 1, 1), out.flip(-1), out)
    elif cfg["flip"] == "random":
        mask = torch.rand(n, device=dev) < 0.5
        out = torch.where(mask.view(-1, 1, 1, 1), out.flip(-1), out)
    c = cfg["cutout"]
    if c > 0:
        cy = torch.randint(0, 32 - c + 1, (n, 1, 1, 1), device=dev)
        cx = torch.randint(0, 32 - c + 1, (n, 1, 1, 1), device=dev)
        ar = torch.arange(32, device=dev)
        my = (ar.view(1, 1, 32, 1) >= cy) & (ar.view(1, 1, 32, 1) < cy + c)
        mx = (ar.view(1, 1, 1, 32) >= cx) & (ar.view(1, 1, 1, 32) < cx + c)
        mean = torch.tensor(CIFAR_MEAN, device=dev, dtype=out.dtype).view(1, 3, 1, 1)
        out = torch.where(my & mx, mean, out)
    return out


def build(context: BuildContext):
    cfg = {**DEFAULTS, **context.parameters}
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = context.device
    model = Net(cfg["widths"], cfg["bn_momentum"], cfg["scaling_factor"], context.num_classes)
    model = model.to(device).to(memory_format=torch.channels_last)
    loss_module = TrainLoss(model, cfg["label_smoothing"])
    if device.type == "cuda":
        compiled = torch.compile(loss_module, mode=cfg["compile_mode"])
    else:
        compiled = loss_module
    state = SimpleNamespace(model=model, compiled=compiled, context=context, cfg=cfg)
    # Warm up compile/autotune on synthetic data; prepare resets everything afterwards.
    bs = cfg["batch_size"]
    x = torch.rand(bs, 3, 32, 32, device=device, dtype=torch.float16)
    y = torch.randint(0, context.num_classes, (bs,), device=device)
    opt = make_optimizer(model, cfg)
    model.train()
    for _ in range(cfg["warmup_steps"]):
        loss = compiled(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        for b in (1, 7, 1024):
            model(torch.rand(b, 3, 32, 32, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return state


def make_optimizer(model, cfg):
    momentum = cfg["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = cfg["lr"] / kilostep_scale
    wd = cfg["weight_decay"] * cfg["batch_size"] / kilostep_scale
    lr_biases = lr * cfg["bias_scaler"]
    whiten_bias = [model.whiten.bias]
    norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
    others = [
        p
        for k, p in model.named_parameters()
        if "norm" not in k and p.requires_grad and p is not model.whiten.bias
    ]
    groups = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases, base_lr=lr_biases),
        dict(params=others, lr=lr, weight_decay=wd / lr, base_lr=lr),
        dict(params=whiten_bias, lr=lr, weight_decay=wd / lr, base_lr=lr, whiten=True),
    ]
    return torch.optim.SGD(groups, momentum=momentum, nesterov=True)


def prepare(state, data: TrainingData, seed: int) -> None:
    cfg = state.cfg
    device = state.context.device
    model = state.model
    reset_model(model)
    model.train()
    images = data.images.to(device, non_blocking=True)
    state.labels = data.labels.to(device, non_blocking=True)
    imgs = images.to(torch.float16).div_(255)
    mean = torch.tensor(CIFAR_MEAN, device=device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device, dtype=torch.float16).view(1, 3, 1, 1)
    init_whitening(model.whiten, (imgs[: cfg["whiten_images"]] - mean) / std)
    r = cfg["translate"]
    state.padded = F.pad(imgs, (r, r, r, r), mode="reflect") if r > 0 else imgs
    state.flip_mask = torch.rand(len(imgs), device=device) < 0.5
    state.optimizer = make_optimizer(model, cfg)
    state.ema = [t for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point()]
    state.ema = [t.detach().clone() for t in state.ema]


def train(state) -> nn.Module:
    cfg = state.cfg
    model = state.model
    opt = state.optimizer
    bs = cfg["batch_size"]
    n = len(state.labels)
    steps_per_epoch = n // bs
    total_steps = math.ceil(cfg["epochs"] * steps_per_epoch)
    warmup = int(total_steps * 0.23)
    alpha = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    live = [t for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point()]
    ema = state.ema
    whiten_steps = cfg["whiten_bias_epochs"] * steps_per_epoch

    def lr_factor(step):
        if step < warmup:
            frac = step / warmup
            return 0.2 * (1 - frac) + 1.0 * frac
        frac = (step - warmup) / (total_steps - warmup)
        return 1.0 * (1 - frac) + 0.07 * frac

    step = 0
    epoch = 0
    model.train()
    while step < total_steps:
        aug = augment_epoch(state.padded, state.flip_mask, epoch, cfg)
        perm = torch.randperm(n, device=aug.device)
        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = perm[i * bs : (i + 1) * bs]
            f = lr_factor(step)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
                if g.get("whiten") and step >= whiten_steps:
                    g["lr"] = 0.0
            loss = state.compiled(aug[idx], state.labels[idx])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                with torch.no_grad():
                    torch._foreach_lerp_(ema, live, 1 - alpha[step].item())
                    torch._foreach_copy_(live, ema)
        epoch += 1
    with torch.no_grad():
        torch._foreach_copy_(live, ema)
    return model
