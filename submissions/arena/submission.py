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
    "epochs": 11,
    "batch_size": 1536,
    "lr": 9.0,
    "momentum": 0.85,
    "weight_decay": 0.012,
    "bias_scaler": 64.0,
    "label_smoothing": 0.3,
    "whiten_bias_epochs": 5,  # afterwards the whitening output is detached
    "translate": 2,
    "widths": (128, 384, 576),
    "convs_per_group": (2, 3, 3),
    "bn_momentum": 0.6,
    "scaling_factor": 1 / 9,
    "compile": True,
    "res_schedule": ((20, 2), (24, 3), (28, 2)),  # (resolution, epochs) stages before full 32 px
    "freeze_first_full_res": True,  # freeze group 1's first conv block in the 32 px epochs
    "optimizer": "muon",  # "sgd" or "muon" (Muon on conv filters, SGD on the rest)
    "muon_lr": 0.205,
    "muon_momentum": 0.655,
    "ns_steps": 2,
    "ns_precond": True,  # AOL preconditioning before Newton-Schulz (Turbo-Muon)
    "compile_mode": "max-autotune",
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

    def forward(self, x, detach_first: bool = False):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            if i == 0:
                x = self.pool(x)
            x = self.activ(norm(x))
            if i == 0 and detach_first:
                x = x.detach()
        return x


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        (n1, n2, n3), m = hyp["convs_per_group"], hyp["bn_momentum"]
        whiten_width = 2 * 3 * 2 * 2
        self.whiten = nn.Conv2d(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, w1, n1, m),
            ConvGroup(w1, w2, n2, m),
            ConvGroup(w2, w3, n3, m),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scale = hyp["scaling_factor"]

    def forward(self, x, detach_whiten: bool = False, freeze_first: bool = False):
        # x: normalized fp16 channels_last images. Once the whitening bias is frozen,
        # detaching its output skips the backward into it at full resolution;
        # freeze_first also stops training the first conv block of group 1.
        x = self.whiten(x)
        if detach_whiten:
            x = x.detach()
        x = self.layers[1](self.layers[0](x), detach_first=freeze_first)
        for layer in self.layers[2:]:
            x = layer(x)
        return self.head(x) * self.scale


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


# ----------------------------------------------------------------------------- muon


def zeropower_via_newtonschulz5(G, steps, precond=False):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    if precond:
        # Turbo-Muon style almost-orthogonal (AOL) rescaling of the rows instead of a
        # Frobenius normalization; the rescaled Gram matrix is reused by the first step.
        A = X @ X.T
        d = A.abs().sum(1).clamp_min(1e-7).rsqrt()
        X = X * d[:, None]
        A = A * d[:, None] * d[None, :]
    else:
        X = X / (X.norm() + 1e-7)
    for i in range(steps):
        if i > 0 or not precond:
            A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def muon_update(params, grads, bufs, lr, momentum: float, ns_steps: int, precond: bool):
    """One Muon step for a list of conv filters; lr is a 0-dim GPU tensor (no recompiles)."""
    for p, g, buf in zip(params, grads, bufs):
        buf.mul_(momentum).add_(g)
        g = g.add(buf, alpha=momentum)
        p.mul_(len(p) ** 0.5 / p.norm())
        update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), ns_steps, precond)
        p.sub_(update.view(g.shape).to(p.dtype) * lr)


class MuonStep:
    """Muon on a fixed list of conv filters, compiled and captured as CUDA graphs.

    Parameters keep their storage across trials (they are reset in place), so the graphs
    built in build() are reused: each step copies the fresh grads into static buffers,
    writes the lr into a GPU scalar and replays the graph for the set of filters that
    received grads (all of them, or one of `subsets` when some are frozen).
    reset() clears the momentum.
    """

    def __init__(
        self, params, momentum, ns_steps, use_graph, compiled=False, precond=False, subsets=()
    ):
        self.precond = precond
        self.update_fn = torch.compile(muon_update, dynamic=False) if compiled else muon_update
        self.params = params
        self.grads = [torch.zeros_like(p) for p in params]
        self.bufs = [torch.zeros_like(p) for p in params]
        self.lr_t = torch.zeros((), device=params[0].device, dtype=torch.float32)
        self.momentum, self.ns_steps = momentum, ns_steps
        self.graphs = {}
        if use_graph:
            for idx in (tuple(range(len(params))), *subsets):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), torch.no_grad():
                    for _ in range(3):
                        self._update(idx)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    self._update(idx)
                self.graphs[idx] = graph

    def _update(self, idx):
        self.update_fn(
            [self.params[i] for i in idx],
            [self.grads[i] for i in idx],
            [self.bufs[i] for i in idx],
            self.lr_t,
            self.momentum,
            self.ns_steps,
            self.precond,
        )

    def reset(self):
        torch._foreach_zero_(self.bufs)

    @torch.no_grad()
    def step(self, lr):
        idx = tuple(i for i, p in enumerate(self.params) if p.grad is not None)
        torch._foreach_copy_([self.grads[i] for i in idx], [self.params[i].grad for i in idx])
        self.lr_t.fill_(lr)
        if idx in self.graphs:
            self.graphs[idx].replay()
        else:
            self._update(idx)


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


def augment_epoch(padded, flip_bits, epoch, translate, scale, shift):
    """Random 32x32 crops from reflect-padded uint8 images plus alternating flip, then
    normalization to fp16 (x * scale + shift). No host sync."""
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
    out = padded[idx_n, idx_c, rows, cols].half().mul_(scale).add_(shift)
    return out.contiguous(memory_format=torch.channels_last)


def downscale(x, size):
    out = F.interpolate(x.float(), size=(size, size), mode="bilinear", antialias=True)
    return out.half().contiguous(memory_format=torch.channels_last)


def train_flags(hyp, epoch, res):
    """(detach the whitening output, freeze group 1's first block) for this epoch."""
    detach_whiten = epoch >= hyp["whiten_bias_epochs"]
    return detach_whiten, detach_whiten and hyp["freeze_first_full_res"] and res is None


def resolution_at(schedule, epoch):
    """Resolution for this epoch from the (resolution, epochs) stages; None = full size."""
    for res, n_epochs in schedule:
        if epoch < n_epochs:
            return res
        epoch -= n_epochs
    return None


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
        train_net = torch.compile(net, mode=hyp["compile_mode"], dynamic=False)
    state = SimpleNamespace(hyp=hyp, context=context, model=model, net=net, train_net=train_net)
    state.muon = None
    if hyp["optimizer"] == "muon":
        filters = [p for p in net.parameters() if p.ndim == 4 and p.requires_grad]
        first = net.layers[1].convs[0].weight
        frozen_subset = tuple(i for i, p in enumerate(filters) if p is not first)
        cuda = device.type == "cuda"
        state.muon = MuonStep(
            filters,
            hyp["muon_momentum"],
            hyp["ns_steps"],
            use_graph=cuda,
            compiled=cuda and hyp["compile"],
            precond=hyp["ns_precond"],
            subsets=(frozen_subset,) if hyp["freeze_first_full_res"] else (),
        )

    # Warm up compilation on synthetic data (untimed); prepare() resets everything after.
    if device.type == "cuda":
        bs = hyp["batch_size"]
        x = torch.randn(bs, 3, 32, 32, device=device).half().contiguous(
            memory_format=torch.channels_last
        )
        y = torch.randint(0, context.num_classes, (bs,), device=device)
        net.train()
        # Warm exactly the (resolution, flags) variants that train() will use.
        variants = {(32, False, False)}
        for e in range(math.ceil(hyp["epochs"])):
            res = resolution_at(hyp["res_schedule"], e)
            variants.add((res or 32, *train_flags(hyp, e, res)))
        for res, detach, freeze in sorted(variants):
            xs = x if res == 32 else downscale(x, res)
            for _ in range(3):
                out = train_net(xs, detach, freeze)
                loss = F.cross_entropy(
                    out, y, label_smoothing=hyp["label_smoothing"], reduction="none"
                ).sum()
                loss.backward()
                for p in net.parameters():
                    p.grad = None
        # Also warm up the prepare/augment path (eigh, pad, gather) on synthetic uint8 data.
        fake = TrainingData(
            torch.randint(0, 256, (6000, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (6000,)),
        )
        prepare(state, fake, 0)
        augment_epoch(state.padded, state.flip_bits, 0, hyp["translate"], state.scale, state.shift)
        # Warm up the optimizer steps (SGD state allocation, Muon graph replay).
        for _ in range(2):
            out = train_net(x)
            F.cross_entropy(out, y, reduction="none").sum().backward()
            state.optimizer.step()
            if state.muon is not None:
                state.muon.step(hyp["muon_lr"])
            net.zero_grad(set_to_none=True)
        state.padded = state.labels = None
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
    state.scale = (1 / (255 * std)).half()
    state.shift = (-mean / std).half()
    subset = images[torch.randperm(len(images), device=device)[:5000]]
    init_whitening_conv(net.whiten, subset.half() * state.scale + state.shift)
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
    if state.muon is not None:
        other = [p for p in other if p.ndim != 4]
        state.muon.reset()
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
    live = [t for t in net.state_dict().values() if t.dtype in (torch.half, torch.float)]
    ema = [t.detach().clone() for t in live]
    ls = hyp["label_smoothing"]

    step = 0
    epoch = 0
    while step < total:
        inputs_all = augment_epoch(
            state.padded, state.flip_bits, epoch, hyp["translate"], state.scale, state.shift
        )
        res = resolution_at(hyp["res_schedule"], epoch)
        if res is not None:
            inputs_all = downscale(inputs_all, res)
        flags = train_flags(hyp, epoch, res)
        perm = torch.randperm(n, device=inputs_all.device)
        if epoch >= hyp["whiten_bias_epochs"]:
            opt.param_groups[0]["base_lr"] = 0.0
        for b in range(steps_per_epoch):
            if step >= total:
                break
            idx = perm[b * bs : (b + 1) * bs]
            out = train_net(inputs_all[idx], *flags)
            loss = F.cross_entropy(out, state.labels[idx], label_smoothing=ls, reduction="none").sum()
            loss.backward()
            f = lr_factor(step, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.step()
            if state.muon is not None:
                state.muon.step(hyp["muon_lr"] * f)
            net.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                decay = alpha[step]
                with torch.no_grad():
                    torch._foreach_lerp_(ema, live, 1 - decay)
                    torch._foreach_copy_(live, ema)
        epoch += 1
    with torch.no_grad():
        torch._foreach_copy_(live, ema)
    state.padded = None
    return state.model
