"""Optimisation: airbench SGD with a triangular schedule and lookahead, and the training loop."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig
from .data import TrainingStream
from .model import Net

Schedule = Callable[[int], float]


def triangle(steps: int, start: float, peak_frac: float, end: float) -> Schedule:
    """Piecewise-linear multiplier rising from ``start`` to 1 at ``peak_frac``, then to ``end``."""
    peak = max(1, int(peak_frac * steps))

    def schedule(i: int) -> float:
        if i < peak:
            return start + (1 - start) * i / peak
        return 1 + (end - 1) * (i - peak) / max(1, steps - peak)

    return schedule


def make_optimizer(
    model: Net, config: RecipeConfig, total_steps: int, steps_per_epoch: int
) -> tuple[torch.optim.SGD, list[tuple[dict, float, Schedule]]]:
    """Nesterov SGD in airbench's parametrisation, with one lr schedule per parameter group.

    lr and weight decay are per 1024 examples and decoupled from momentum. BatchNorm biases
    take ``bias_scaler`` times the learning rate but the same decay strength. The whitening
    bias trains for ``whiten_bias_epochs`` and is then frozen by a zero learning rate rather
    than ``requires_grad``, so the autograd and compiled graphs never change mid-run.
    """
    norm_ids = {id(m.bias) for m in model.modules() if isinstance(m, nn.BatchNorm2d)}
    whiten = model.whiten.bias
    params = [p for p in model.parameters() if p.requires_grad]
    norms = [p for p in params if id(p) in norm_ids]
    others = [p for p in params if id(p) not in norm_ids and p is not whiten]
    kilostep = 1024 * (1 + 1 / (1 - config.momentum))
    lr = config.lr / kilostep
    wd = config.weight_decay * config.batch_size / kilostep
    lr_bias = lr * config.bias_scaler
    groups = [
        {"params": norms, "lr": lr_bias, "weight_decay": wd / lr_bias},
        {"params": others, "lr": lr, "weight_decay": wd / lr},
        {"params": [whiten], "lr": lr, "weight_decay": wd / lr},
    ]
    optimizer = torch.optim.SGD(
        groups, momentum=config.momentum, nesterov=True, fused=config.fused_sgd or None
    )
    schedule = triangle(total_steps, config.lr_start, config.lr_peak_frac, config.lr_end)
    whiten_steps = ceil(config.whiten_bias_epochs * steps_per_epoch)

    def whiten_schedule(i: int) -> float:
        return schedule(i) if i < whiten_steps else 0.0

    norm_group, other_group, whiten_group = optimizer.param_groups
    schedules = [
        (norm_group, norm_group["lr"], schedule),
        (other_group, other_group["lr"], schedule),
        (whiten_group, whiten_group["lr"], whiten_schedule),
    ]
    return optimizer, schedules


class Lookahead:
    """airbench lookahead: every few steps, pull weights toward a slow EMA and copy it back."""

    def __init__(self, model: Net) -> None:
        floating = (torch.half, torch.float)
        self.current = [v for v in model.state_dict().values() if v.dtype in floating]
        self.slow = [v.detach().clone() for v in self.current]

    @torch.no_grad()
    def update(self, decay: float) -> None:
        torch._foreach_lerp_(self.slow, self.current, 1 - decay)
        torch._foreach_copy_(self.current, self.slow)


def resolution(config: RecipeConfig, step: int, total_steps: int) -> int | None:
    """Training resolution in force at ``step`` under ``res_schedule``."""
    size = None
    for start, value in config.res_schedule:
        if step >= start * total_steps:
            size = value
    return size


def fit(model: Net, step_model: nn.Module, stream: TrainingStream, config: RecipeConfig) -> None:
    """Train for ``config.epochs`` epochs (a fractional final epoch stops mid-epoch)."""
    total_steps = ceil(stream.steps_per_epoch * config.epochs)
    optimizer, schedules = make_optimizer(model, config, total_steps, stream.steps_per_epoch)
    lookahead = Lookahead(model) if config.lookahead else None
    model.train()
    step = 0
    for epoch in range(ceil(config.epochs)):
        for inputs, labels in stream.epoch(epoch):
            size = resolution(config, step, total_steps)
            if size is not None and size != inputs.size(-1):
                inputs = F.interpolate(
                    inputs, size=(size, size), mode="bilinear", align_corners=False, antialias=True
                )
            for group, base_lr, schedule in schedules:
                group["lr"] = base_lr * schedule(step)
            outputs = step_model(inputs)
            loss = F.cross_entropy(
                outputs, labels, label_smoothing=config.label_smoothing, reduction="sum"
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if lookahead is not None and step % 5 == 0:
                lookahead.update(decay=0.95**5 * (step / total_steps) ** 3)
            if step >= total_steps:
                break
    if lookahead is not None:
        lookahead.update(decay=1.0)
