"""Optimisation: airbench SGD (triangular schedule, lookahead) or Muon, and the training loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig
from .data import TrainingStream
from .model import AirbenchNet

Schedule = Callable[[int], float]


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise a matrix with a quintic Newton-Schulz iteration (Muon)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon for conv filters as in airbench94_muon: Nesterov momentum, orthogonalised update,
    and per-step renormalisation of each filter bank to norm sqrt(out_channels)."""

    def __init__(self, params, lr: float, momentum: float, ns_steps: int) -> None:
        super().__init__(params, {"lr": lr, "momentum": momentum, "ns_steps": ns_steps})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), group["ns_steps"])
                p.add_(update.view(g.shape).to(p.dtype), alpha=-group["lr"])


@dataclass
class Optimisation:
    """Optimisers plus one learning-rate multiplier schedule per parameter group."""

    optimizers: list[torch.optim.Optimizer]
    schedules: list[tuple[dict, float, Schedule]]

    def set_lr(self, step: int) -> None:
        for group, base_lr, schedule in self.schedules:
            group["lr"] = base_lr * schedule(step)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)


def triangle(steps: int, start: float, peak_frac: float, end: float) -> Schedule:
    """Piecewise-linear multiplier rising from ``start`` to 1 at ``peak_frac``, then to ``end``."""
    peak = max(1, int(peak_frac * steps))

    def schedule(i: int) -> float:
        if i < peak:
            return start + (1 - start) * i / peak
        return 1 + (end - 1) * (i - peak) / max(1, steps - peak)

    return schedule


def linear_decay(steps: int) -> Schedule:
    return lambda i: max(0.0, 1 - i / steps)


def _split(model: nn.Module) -> tuple[list, list, list]:
    """(BatchNorm biases, whitening bias, all other trainable parameters)."""
    norm_ids = {
        id(m.bias)
        for m in model.modules()
        if isinstance(m, nn.BatchNorm2d) and m.bias.requires_grad
    }
    whiten = [model.whiten.bias] if isinstance(model, AirbenchNet) else []
    whiten_ids = {id(p) for p in whiten}
    params = [p for p in model.parameters() if p.requires_grad]
    norms = [p for p in params if id(p) in norm_ids]
    others = [p for p in params if id(p) not in norm_ids | whiten_ids]
    return norms, whiten, others


def make_optimisation(
    model: nn.Module, config: RecipeConfig, total_steps: int, steps_per_epoch: int
) -> Optimisation:
    norms, whiten, others = _split(model)
    if config.optimizer == "sgd":
        # airbench parametrisation: lr and wd per 1024 examples, decoupled from momentum.
        kilostep = 1024 * (1 + 1 / (1 - config.momentum))
        lr = config.lr / kilostep
        wd = config.weight_decay * config.batch_size / kilostep
        lr_bias = lr * config.bias_scaler
        groups = [
            {"params": norms, "lr": lr_bias, "weight_decay": wd / lr_bias},
            {"params": whiten + others, "lr": lr, "weight_decay": wd / lr},
        ]
        sgd = torch.optim.SGD(groups, momentum=config.momentum, nesterov=True)
        schedule = triangle(total_steps, config.lr_start, config.lr_peak_frac, config.lr_end)
        return Optimisation([sgd], [(g, g["lr"], schedule) for g in sgd.param_groups])

    # Muon for conv filters; SGD for biases and the head (airbench94_muon).
    filters = [p for p in others if p.ndim == 4]
    rest = [p for p in others if p.ndim != 4]
    wd = config.muon_weight_decay * config.batch_size
    groups = [
        {"params": whiten, "lr": config.muon_bias_lr},
        {"params": norms, "lr": config.muon_bias_lr},
        {"params": rest, "lr": config.muon_head_lr},
    ]
    groups = [g | {"weight_decay": wd / g["lr"]} for g in groups if g["params"]]
    sgd = torch.optim.SGD(groups, momentum=config.momentum, nesterov=True)
    muon = Muon(filters, config.muon_lr, config.muon_momentum, config.muon_ns_steps)
    whiten_steps = max(1, ceil(config.whiten_bias_epochs * steps_per_epoch))
    schedules = []
    for group in sgd.param_groups:
        is_whiten = whiten and group["params"][0] is whiten[0]
        decay = linear_decay(whiten_steps if is_whiten else total_steps)
        schedules.append((group, group["lr"], decay))
    schedules += [(g, g["lr"], linear_decay(total_steps)) for g in muon.param_groups]
    return Optimisation([sgd, muon], schedules)


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


def resolution(config: RecipeConfig, step: int, total_steps: int) -> int | None:
    """Training resolution at ``step`` from ``res_schedule`` [(start_fraction, size), ...]."""
    size = None
    for start, value in config.res_schedule:
        if step >= start * total_steps:
            size = value
    return size


def fit(
    model: nn.Module, step_model: nn.Module, stream: TrainingStream, config: RecipeConfig
) -> None:
    """Train for ``config.epochs`` epochs (fractional epochs stop mid-epoch)."""
    total_steps = ceil(stream.steps_per_epoch * config.epochs)
    plan = make_optimisation(model, config, total_steps, stream.steps_per_epoch)
    lookahead = Lookahead(model) if config.lookahead else None
    whiten_bias = model.whiten.bias if isinstance(model, AirbenchNet) else None
    model.train()
    step = 0
    for epoch in range(ceil(config.epochs)):
        if whiten_bias is not None and config.optimizer == "sgd":
            whiten_bias.requires_grad = epoch < config.whiten_bias_epochs
        for inputs, labels in stream.epoch(epoch):
            size = resolution(config, step, total_steps)
            if size is not None and size != inputs.size(-1):
                inputs = F.interpolate(
                    inputs, size=(size, size), mode="bilinear", align_corners=False, antialias=True
                )
            plan.set_lr(step)
            outputs = step_model(inputs)
            loss = F.cross_entropy(
                outputs, labels, label_smoothing=config.label_smoothing, reduction="sum"
            )
            loss.backward()
            plan.step()
            plan.zero_grad()
            step += 1
            if lookahead is not None and step % 5 == 0:
                lookahead.update(model, decay=0.95**5 * (step / total_steps) ** 3)
            if step >= total_steps:
                break
    if lookahead is not None:
        lookahead.update(model, decay=1.0)
