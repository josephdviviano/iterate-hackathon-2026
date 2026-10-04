"""Fast CIFAR-100 recipe: airbench-style convnet, GPU-resident data and augmentation."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

DEFAULTS = dict(
    widths=(128, 384, 512),
    epochs=16,
    batch_size=512,
    lr=0.4,
    momentum=0.9,
    weight_decay=5e-4,
    label_smoothing=0.2,
    warmup_frac=0.2,
    translate=2,
    whiten_patches=5000,
    scale=1 / 9,
    compile=True,
)


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn1(x))
        x = F.gelu(self.bn2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths, num_classes, scale):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad_(False)
        c = 24
        groups = []
        for w in widths:
            groups.append(ConvGroup(c, w))
            c = w
        self.groups = nn.Sequential(*groups)
        self.head = nn.Linear(c, num_classes, bias=False)
        self.scale = scale

    def features(self, x):
        x = F.gelu(self.whiten(x))
        x = self.groups(x)
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.head(x) * self.scale

    def forward(self, x):
        # Evaluation convention: float32 RGB in [0, 1]; normalization lives here.
        x = (x - self.mean) / self.std
        return self.features(x.contiguous(memory_format=torch.channels_last))


def whitening_weights(patches, eps=5e-4):
    n, c, h, w = patches.shape
    flat = patches.reshape(n, -1).float()
    flat = flat - flat.mean(0)
    cov = flat.T @ flat / n
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals, eigvecs = eigvals.flip(0), eigvecs.T.flip(0)
    filters = (eigvecs / torch.sqrt(eigvals[:, None] + eps)).reshape(-1, c, h, w)
    return torch.cat([filters, -filters])


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()  # also resets running stats
    model.head.weight.data.zero_()


def augment(padded, translate, generator=None):
    """Random flips and translations of a padded image batch, fully on GPU."""
    n = padded.shape[0]
    dev = padded.device
    span = 2 * translate + 1
    sy = torch.randint(0, span, (n,), device=dev)
    sx = torch.randint(0, span, (n,), device=dev)
    base = torch.arange(32, device=dev)
    rows = (sy[:, None] + base)[:, None, :, None]
    cols = sx[:, None] + base
    flip = torch.rand(n, device=dev) < 0.5
    cols = torch.where(flip[:, None], cols.flip(1), cols)[:, None, None, :]
    idx = torch.arange(n, device=dev)[:, None, None, None]
    ch = torch.arange(3, device=dev)[None, :, None, None]
    return padded[idx, ch, rows, cols].contiguous(memory_format=torch.channels_last)


def make_optimizer(state):
    hp = state.hp
    decay, no_decay = [], []
    for name, p in state.model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.SGD(
        [
            dict(params=decay, weight_decay=hp["weight_decay"]),
            dict(params=no_decay, weight_decay=0.0),
        ],
        lr=hp["lr"],
        momentum=hp["momentum"],
        nesterov=True,
    )


def train_step(state, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = state.step_model(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=state.hp["label_smoothing"])
    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    state.optimizer.step()


def build(context: BuildContext):
    hp = {**DEFAULTS, **context.parameters}
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = Net(hp["widths"], context.num_classes, hp["scale"]).to(context.device)
    model = model.to(memory_format=torch.channels_last)
    step_model = torch.compile(model.features) if hp["compile"] else model.features
    state = SimpleNamespace(model=model, step_model=step_model, context=context, hp=hp)
    if context.device.type == "cuda":
        # Untimed warmup on synthetic inputs: compile and autotune the training step.
        bs = hp["batch_size"]
        state.optimizer = make_optimizer(state)
        x = torch.randn(bs, 3, 32, 32, device=context.device).contiguous(
            memory_format=torch.channels_last
        )
        y = torch.randint(0, context.num_classes, (bs,), device=context.device)
        model.train()
        for _ in range(3):
            train_step(state, x, y)
        padded = torch.zeros(1000, 3, 36, 36, device=context.device, dtype=torch.bfloat16)
        augment(padded, hp["translate"])
        whitening_weights(torch.randn(1000, 3, 2, 2, device=context.device))
        torch.cuda.synchronize()
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    hp, dev = state.hp, state.context.device
    model = state.model
    reset_model(model)
    model.train()
    images = data.images.to(dev, non_blocking=True).float().div_(255)
    mean = images.mean(dim=(0, 2, 3), keepdim=True)
    std = images.std(dim=(0, 2, 3), keepdim=True)
    model.mean.copy_(mean)
    model.std.copy_(std)
    images = ((images - mean) / std).to(torch.bfloat16)
    patches = images[: hp["whiten_patches"]].float().unfold(2, 2, 1).unfold(3, 2, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 2, 2)
    model.whiten.weight.data.copy_(whitening_weights(patches))
    t = hp["translate"]
    state.padded = F.pad(images, (t, t, t, t), mode="reflect") if t else images
    state.labels = data.labels.to(dev, non_blocking=True)
    state.optimizer = make_optimizer(state)


def train(state) -> nn.Module:
    hp = state.hp
    n = state.labels.shape[0]
    bs = hp["batch_size"]
    steps_per_epoch = n // bs
    total = hp["epochs"] * steps_per_epoch
    warm = max(1, int(hp["warmup_frac"] * total))
    step = 0
    for epoch in range(hp["epochs"]):
        images = augment(state.padded, hp["translate"])
        perm = torch.randperm(n, device=images.device)
        for i in range(steps_per_epoch):
            idx = perm[i * bs : (i + 1) * bs]
            lr = hp["lr"] * (step / warm if step < warm else (total - step) / (total - warm))
            for g in state.optimizer.param_groups:
                g["lr"] = lr
            train_step(state, images[idx], state.labels[idx])
            step += 1
    state.model.eval()
    return state.model
