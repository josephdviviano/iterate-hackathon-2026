"""Optimisation: decoupled Nesterov SGD, triangular learning-rate schedule and lookahead."""

from __future__ import annotations

from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig
from .data import TrainingStream
from .model import AirbenchNet


def make_optimizer(model: nn.Module, config: RecipeConfig) -> torch.optim.SGD:
    """SGD whose lr and weight decay are decoupled from momentum (airbench parametrisation).

    BatchNorm biases take ``bias_scaler`` times the learning rate but the same decay strength.
    """
    kilostep_scale = 1024 * (1 + 1 / (1 - config.momentum))
    lr = config.lr / kilostep_scale
    wd = config.weight_decay * config.batch_size / kilostep_scale
    lr_bias = lr * config.bias_scaler
    norm_biases = {
        id(m.bias)
        for m in model.modules()
        if isinstance(m, nn.BatchNorm2d) and m.bias.requires_grad
    }
    params = [p for p in model.parameters() if p.requires_grad]
    biases = [p for p in params if id(p) in norm_biases]
    others = [p for p in params if id(p) not in norm_biases]
    groups = [
        {"params": biases, "lr": lr_bias, "weight_decay": wd / lr_bias},
        {"params": others, "lr": lr, "weight_decay": wd / lr},
    ]
    return torch.optim.SGD(groups, momentum=config.momentum, nesterov=True)


def triangle(steps: int, start: float, peak_frac: float, end: float) -> list[float]:
    """Piecewise-linear multiplier rising from ``start`` to 1 at ``peak_frac``, then to ``end``."""
    peak = max(1, int(peak_frac * steps))
    rise = [start + (1 - start) * i / peak for i in range(peak)]
    fall = [1 + (end - 1) * (i - peak) / max(1, steps - peak) for i in range(peak, steps + 1)]
    return rise + fall


class Lookahead:
    """airbench lookahead: every few steps, pull weights toward a slow EMA and copy it back."""

    def __init__(self, model: nn.Module) -> None:
        self.slow = [v.detach().clone() for v in model.state_dict().values()]

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float) -> None:
        for slow, current in zip(self.slow, model.state_dict().values(), strict=True):
            if current.dtype in (torch.half, torch.float):
                slow.lerp_(current, 1 - decay)
                current.copy_(slow)


def fit(
    model: nn.Module,
    step_model: nn.Module,
    stream: TrainingStream,
    optimizer: torch.optim.SGD,
    config: RecipeConfig,
) -> None:
    """Train for ``config.epochs`` epochs (fractional epochs stop mid-epoch)."""
    total_steps = ceil(stream.steps_per_epoch * config.epochs)
    schedule = triangle(total_steps, config.lr_start, config.lr_peak_frac, config.lr_end)
    alpha = [0.95**5 * (i / total_steps) ** 3 for i in range(total_steps + 1)]
    lookahead = Lookahead(model) if config.lookahead else None
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    whiten_bias = model.whiten.bias if isinstance(model, AirbenchNet) else None
    model.train()
    step = 0
    for epoch in range(ceil(config.epochs)):
        if whiten_bias is not None:
            whiten_bias.requires_grad = epoch < config.whiten_bias_epochs
        for inputs, labels in stream.epoch(epoch):
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr * schedule[step]
            outputs = step_model(inputs)
            loss = F.cross_entropy(
                outputs, labels, label_smoothing=config.label_smoothing, reduction="sum"
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if lookahead is not None and step % 5 == 0:
                lookahead.update(model, decay=alpha[step])
            if step >= total_steps:
                break
    if lookahead is not None:
        lookahead.update(model, decay=1.0)
