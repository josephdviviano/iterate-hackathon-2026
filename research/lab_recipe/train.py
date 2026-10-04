"""Optimisation: airbench SGD (triangular schedule, lookahead) or Muon, and the training loop."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from . import narrowing
from .config import RecipeConfig
from .data import TrainingStream, batch_crop
from .model import AirbenchNet

Schedule = Callable[[int], float]


NS_COEFFICIENTS = {"airbench": (3.4445, -4.7750, 2.0315), "hiverge": (3.4576, -4.7391, 2.0843)}


def zeropower_via_newtonschulz5(
    g: torch.Tensor, steps: int, eps: float = 1e-7, coefficients: str = "airbench"
) -> torch.Tensor:
    """Approximately orthogonalise a matrix with a quintic Newton-Schulz iteration (Muon)."""
    a, b, c = NS_COEFFICIENTS[coefficients]
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

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        ns_steps: int,
        coefficients: str = "airbench",
        renorm: str = "every",
        weight_decay: float = 0.0,
        total_steps: int = 1,
    ) -> None:
        super().__init__(params, {"lr": lr, "momentum": momentum, "ns_steps": ns_steps})
        self.coefficients = coefficients
        self.renorm = renorm
        self.decoupled_wd = weight_decay
        self.total_steps = total_steps
        self.step_count = 0
        self.last_norm_step = 0

    @torch.no_grad()
    def step(self) -> None:
        # hiverge renormalises filters every 2 + int(15 * progress) steps, not every step.
        if self.renorm == "hiverge":
            cadence = 2 + int(15 * self.step_count / self.total_steps)
            do_norm = self.step_count - self.last_norm_step >= cadence
            if do_norm:
                self.last_norm_step = self.step_count
        else:
            do_norm = True
        self.step_count += 1
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
                if do_norm:
                    p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower_via_newtonschulz5(
                    g.reshape(len(g), -1), group["ns_steps"], coefficients=self.coefficients
                )
                p.add_(update.view(g.shape).to(p.dtype), alpha=-group["lr"])
                if self.decoupled_wd:
                    p.mul_(1 - group["lr"] * self.decoupled_wd)


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


def lr_schedule(config: RecipeConfig, steps: int) -> Schedule:
    if config.lr_decay_end < 1 and config.lr_shape == "triangle":
        # The decay reaches lr_end at ``lr_decay_end`` and is then held (a floor tail).
        decay_steps = max(2, round(config.lr_decay_end * steps))
        inner = triangle(decay_steps, config.lr_start, config.lr_peak_frac, config.lr_end)
        return lambda i: inner(min(i, decay_steps))
    return _lr_schedule(config, steps)


def _lr_schedule(config: RecipeConfig, steps: int) -> Schedule:
    """Triangular (default), warmup-stable-decay, or warmup-cosine multiplier."""
    if config.lr_shape == "triangle":
        return triangle(steps, config.lr_start, config.lr_peak_frac, config.lr_end)
    peak = max(1, int(config.lr_peak_frac * steps))
    decay_start = max(peak, int(config.lr_decay_start * steps))

    def schedule(i: int) -> float:
        if i < peak:
            return config.lr_start + (1 - config.lr_start) * i / peak
        if config.lr_shape in ("wsd", "wsd_sqrt"):
            if i < decay_start:
                return 1.0
            frac = (i - decay_start) / max(1, steps - decay_start)
            if config.lr_shape == "wsd_sqrt":  # 1 - sqrt cooldown (Hagele et al. 2024)
                frac = frac**0.5
            return 1 + (config.lr_end - 1) * frac
        frac = (i - peak) / max(1, steps - peak)
        return config.lr_end + (1 - config.lr_end) * 0.5 * (1 + math.cos(math.pi * frac))

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
    whiten = [m.whiten.bias for m in model.modules() if isinstance(m, AirbenchNet)]
    whiten_ids = {id(p) for p in whiten}
    params = [p for p in model.parameters() if p.requires_grad]
    norms = [p for p in params if id(p) in norm_ids]
    others = [p for p in params if id(p) not in norm_ids | whiten_ids]
    return norms, whiten, others


class MasterWeights:
    """fp32 master copies of the fp16 parameters: the optimiser and lookahead update the
    masters, and the fp16 model weights are rounded from them after every update."""

    def __init__(self, model: nn.Module) -> None:
        self.params = [p for p in model.parameters() if p.requires_grad and p.dtype == torch.half]
        self.masters = [nn.Parameter(p.detach().float()) for p in self.params]
        self.by_ptr = {p.data_ptr(): m for p, m in zip(self.params, self.masters, strict=True)}

    def of(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.by_ptr.get(tensor.data_ptr(), tensor)

    def load_grads(self) -> None:
        for p, m in zip(self.params, self.masters, strict=True):
            m.grad = None if p.grad is None else p.grad.float()
            p.grad = None

    @torch.no_grad()
    def sync(self) -> None:
        for p, m in zip(self.params, self.masters, strict=True):
            p.copy_(m)


def make_optimisation(
    model: nn.Module,
    config: RecipeConfig,
    total_steps: int,
    steps_per_epoch: int,
    masters: MasterWeights | None = None,
    whiten_steps: int | None = None,
) -> Optimisation:
    norms, whiten, others = _split(model)
    head: list = []
    multipliers = (config.head_lr_mult, config.head_wd_mult, config.conv_wd_mult)
    if multipliers != (1.0, 1.0, 1.0) and isinstance(model, AirbenchNet):
        head_ids = {id(p) for p in model.head.parameters()}
        head = [p for p in others if id(p) in head_ids]
        others = [p for p in others if id(p) not in head_ids]
    norms1: list = []
    others1: list = []
    if config.stage1_cooldown and isinstance(model, AirbenchNet):
        stage1 = {id(p) for p in model.groups[0].parameters()}
        norms1 = [p for p in norms if id(p) in stage1]
        others1 = [p for p in others if id(p) in stage1]
        norms = [p for p in norms if id(p) not in stage1]
        others = [p for p in others if id(p) not in stage1]
    norms2: list = []
    others2: list = []
    if config.stage2_cooldown and isinstance(model, AirbenchNet):
        stage2 = {id(p) for p in model.groups[1].parameters()}
        norms2 = [p for p in norms if id(p) in stage2]
        others2 = [p for p in others if id(p) in stage2]
        norms = [p for p in norms if id(p) not in stage2]
        others = [p for p in others if id(p) not in stage2]
    # Weight-only freezes: chosen conv weights get their own cooldown, then leave autograd in
    # ``fit``; the stage's BN biases keep training.
    frozen_weights = []
    for entry in config.weight_freeze if isinstance(model, AirbenchNet) else ():
        params = weight_freeze_params(model, entry)
        ids = {id(p) for p in params}
        others, others1, others2 = (
            [p for p in g if id(p) not in ids] for g in (others, others1, others2)
        )
        frozen_weights.append((entry, params))
    if masters is not None:
        norms, whiten, others, head, norms1, others1, norms2, others2 = (
            [masters.of(p) for p in group]
            for group in (norms, whiten, others, head, norms1, others1, norms2, others2)
        )
    if config.optimizer == "sgd":
        # airbench parametrisation: lr and wd per 1024 examples, decoupled from momentum.
        kilostep = 1024 * (1 + 1 / (1 - config.momentum))
        lr = config.lr / kilostep
        wd = config.weight_decay * config.batch_size / kilostep
        lr_bias = lr * config.bias_scaler
        lr_head = lr * config.head_lr_mult
        # BN-bias decay strength (decoupled from the scaler) and stage-1 lr multiplier.
        wd_bias = wd * config.bias_wd_mult
        lr1, lr_bias1 = lr * config.stage1_lr_mult, lr_bias * config.stage1_lr_mult
        schedule = lr_schedule(config, total_steps)
        if whiten_steps is None:
            whiten_steps = ceil(config.whiten_bias_epochs * steps_per_epoch)

        def whiten_schedule(i: int) -> float:
            # Freeze the whitening bias with a zero lr (``fit`` may also drop it from autograd).
            return schedule(i) if i < whiten_steps else 0.0

        def cooldown(fractions: tuple[float, float] | None) -> Schedule:
            # FreezeOut-style: a stage's lr ramps linearly to 0 between the two fractions.
            if not fractions:
                return schedule
            start, end = (f * total_steps for f in fractions)
            return lambda i: schedule(i) * min(1.0, max(0.0, (end - i) / max(1.0, end - start)))

        cooled = cooldown(config.stage1_cooldown)
        cooled2 = cooldown(config.stage2_cooldown)

        groups = [
            (
                {"params": norms, "lr": lr_bias, "weight_decay": wd_bias / lr_bias, "kind": "norm"},
                schedule,
            ),
            ({"params": others, "lr": lr, "weight_decay": wd * config.conv_wd_mult / lr}, schedule),
            ({"params": whiten, "lr": lr, "weight_decay": wd / lr}, whiten_schedule),
            (
                {"params": head, "lr": lr_head, "weight_decay": wd * config.head_wd_mult / lr_head},
                schedule,
            ),
            (
                {
                    "params": norms1,
                    "lr": lr_bias1,
                    "weight_decay": wd_bias / lr_bias1,
                    "kind": "norm",
                },
                cooled,
            ),
            (
                {"params": others1, "lr": lr1, "weight_decay": wd * config.conv_wd_mult / lr1},
                cooled,
            ),
            (
                {
                    "params": norms2,
                    "lr": lr_bias,
                    "weight_decay": wd_bias / lr_bias,
                    "kind": "norm",
                },
                cooled2,
            ),
            ({"params": others2, "lr": lr, "weight_decay": wd * config.conv_wd_mult / lr}, cooled2),
            *(
                (
                    {"params": params, "lr": lr, "weight_decay": wd * config.conv_wd_mult / lr},
                    cooldown((entry[1], entry[2])),
                )
                for entry, params in frozen_weights
            ),
        ]
        groups = [(g, s) for g, s in groups if g["params"]]
        sgd = torch.optim.SGD(
            [g for g, _ in groups],
            momentum=config.momentum,
            nesterov=True,
            fused=config.fused_sgd or None,
        )
        schedules = [
            (group, group["lr"], s) for group, (_, s) in zip(sgd.param_groups, groups, strict=True)
        ]
        return Optimisation([sgd], schedules)

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
    muon = Muon(
        filters,
        config.muon_lr,
        config.muon_momentum,
        config.muon_ns_steps,
        coefficients=config.muon_coefficients,
        renorm=config.muon_renorm,
        weight_decay=config.muon_decoupled_wd * config.batch_size,
        total_steps=total_steps,
    )
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

    def __init__(
        self,
        model: nn.Module,
        masters: MasterWeights | None = None,
        outer_momentum: float = 0.0,
        exclude: set[int] | None = None,
    ) -> None:
        floating = (torch.half, torch.float)
        self.current = [v for v in model.state_dict().values() if v.dtype in floating]
        if masters is not None:
            self.current = [masters.of(v) for v in self.current]
        self.slow = [v.detach().clone() for v in self.current]
        # SNOO (Kallusky et al. 2025): Nesterov momentum on the outer step, for trainable
        # parameters only (not BN statistics, and not stages that freeze, so freezes stay exact).
        self.momentum = outer_momentum
        self.outer = []
        if outer_momentum:
            exclude = exclude or set()
            params = {p.data_ptr() for p in model.parameters() if p.requires_grad}
            params -= exclude
            self.outer = [i for i, v in enumerate(self.current) if v.data_ptr() in params]
            self.buffers = [torch.zeros_like(self.current[i]) for i in self.outer]

    @torch.no_grad()
    def update(self, decay: float) -> None:
        if self.momentum and self.outer and decay < 1:
            current = [self.current[i] for i in self.outer]
            slow = [self.slow[i] for i in self.outer]
            delta = torch._foreach_sub(current, slow)
            torch._foreach_mul_(self.buffers, self.momentum)
            torch._foreach_add_(self.buffers, delta)
            torch._foreach_add_(delta, self.buffers, alpha=self.momentum)
            torch._foreach_add_(slow, delta, alpha=1 - decay)
            rest = set(self.outer)
            others = [i for i in range(len(self.current)) if i not in rest]
            torch._foreach_lerp_(
                [self.slow[i] for i in others], [self.current[i] for i in others], 1 - decay
            )
        else:
            torch._foreach_lerp_(self.slow, self.current, 1 - decay)
        torch._foreach_copy_(self.current, self.slow)


def scheduled(entries: tuple, step: int, total_steps: int, default: int | None) -> int | None:
    """Value in force at ``step`` from ((start_fraction, value), ...) entries."""
    value = default
    for start, entry in entries:
        if step >= start * total_steps:
            value = entry
    return value


class Selector:
    """Scores each batch with a small net and keeps the highest-loss fraction (online form of
    airbench96_faster's proxy masks, which never depend on the main net)."""

    def __init__(
        self, net: nn.Module, config: RecipeConfig, total_steps: int, steps_per_epoch: int
    ) -> None:
        self.net = net
        self.config = config
        self.keep = int(config.batch_size * config.select_fraction)
        selector_config = config.selector_config()
        self.plan = make_optimisation(net, selector_config, total_steps, steps_per_epoch)
        net.train()

    def select(
        self, inputs: torch.Tensor, labels: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update = step % self.config.selector_update_every == 0
        with torch.set_grad_enabled(update):
            losses = F.cross_entropy(
                self.net(inputs),
                labels,
                label_smoothing=self.config.label_smoothing,
                reduction="none",
            )
        keep = losses.detach().topk(self.keep).indices
        if update:
            self.plan.set_lr(step)
            losses[keep].sum().backward()
            self.plan.step()
            self.plan.zero_grad()
        return inputs[keep], labels[keep]


def fit(
    model: nn.Module,
    step_model: nn.Module | list[nn.Module],
    stream: TrainingStream,
    config: RecipeConfig,
    selector_net: nn.Module | None = None,
    narrow: tuple[nn.Module, nn.Module] | None = None,
) -> list[nn.Module]:
    """Train for ``config.epochs`` epochs (fractional epochs stop mid-epoch).

    With ``narrow`` (eager, step) the run continues on that narrower net from the final
    resolution switch (``switch_widths``); it then holds the trained weights."""
    total_steps = stream.steps_until(config.epochs)
    airbench = isinstance(model, AirbenchNet)
    masters = MasterWeights(model) if config.master_fp32 and airbench else None
    plan = make_optimisation(
        model,
        config,
        total_steps,
        stream.steps_per_epoch,
        masters,
        whiten_steps=stream.steps_until(config.whiten_bias_epochs),
    )
    selector = (
        Selector(selector_net, config, total_steps, stream.steps_per_epoch)
        if selector_net is not None and config.select_fraction < 1
        else None
    )
    lookahead = None
    if config.lookahead:
        frozen_ids: set[int] = set()
        if airbench:
            frozen_ids.add(model.whiten.bias.data_ptr())
            for stage, cooldown in ((0, config.stage1_cooldown), (1, config.stage2_cooldown)):
                if cooldown:
                    frozen_ids |= {p.data_ptr() for p in model.groups[stage].parameters()}
        lookahead = Lookahead(model, masters, config.lookahead_outer_momentum, frozen_ids)
    whiten_steps = stream.steps_until(config.whiten_bias_epochs)
    final_size = config.res_schedule[-1][1] if config.res_schedule else None
    switch_step = config.res_schedule[-1][0] * total_steps if config.res_schedule else 0.0
    switched = False
    nets = [model] if narrow is None else [model, narrow[0]]
    saliency = narrowing.Saliency(model, config.narrow_saliency) if narrow else None
    targets = SoftTargets(config, stream, total_steps)
    snapshot_steps = {round(f * total_steps) for f in config.snapshot_fracs}
    snapshots: list[nn.Module] = []
    freezable = isinstance(model, AirbenchNet)
    deepening = freezable and config.residual_start > 0
    model.train()
    step = 0
    crop_generator = None
    if config.crop_schedule:
        # A separate generator (seeded from the harness-seeded global RNG) keeps the data order
        # identical to the uncropped control.
        seed = int(torch.randint(2**31 - 1, (1,)).item())
        crop_generator = torch.Generator(device=stream.labels.device).manual_seed(seed)
    stop = total_steps
    if (
        config.trim_tail
        and config.lookahead
        and not config.lookahead_flush
        and config.lookahead_final_decay == 1.0
    ):
        stop = total_steps - total_steps % config.lookahead_every
    epoch, epoch_start = 0, -1
    # Pruned epochs are shorter, so run epochs until the step budget is spent.
    while step < stop:
        epoch += 1
        if step == epoch_start:
            raise RuntimeError("an epoch yielded no batches (prune_frac too large for batch_size)")
        epoch_start = step
        targets.new_epoch()
        for inputs, labels in stream.epoch(epoch - 1):
            if config.bn_freeze_frac is not None and step == ceil(
                config.bn_freeze_frac * total_steps
            ):
                # FrozenBN tail (Wu & Johnson 2021): BN uses running statistics from here.
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            if freezable:
                model.frozen_groups = scheduled(config.freeze_schedule, step, total_steps, 0)
                if config.thin_window:
                    # Update thinning: inside the window the chosen stages update only every
                    # ``thin_period`` steps and otherwise run in the frozen (no-autograd) graph.
                    a, b = (f * total_steps for f in config.thin_window)
                    if a <= step < b and step % config.thin_period:
                        model.frozen_groups = max(model.frozen_groups, config.thin_groups)
                for entry in config.weight_freeze:
                    if step == ceil(entry[2] * total_steps):
                        for p in weight_freeze_params(model, entry):
                            p.requires_grad_(False)
            size = scheduled(config.res_schedule, step, total_steps, None)
            size = blended_size(config, step, switch_step, size)
            if size == final_size and not switched:
                switched = True
                if narrow is not None:
                    model, step_model, plan, lookahead = _narrow(
                        model, narrow, plan, lookahead, saliency, config, stream, total_steps
                    )
                if step and config.switch_momentum_scale != 1:
                    scale_momentum(plan, config.switch_momentum_scale)
                if step and config.bias_scaler_final is not None:
                    ratio = config.bias_scaler_final / config.bias_scaler
                    plan.schedules = [
                        (g, base * ratio if g.get("kind") == "norm" else base, s)
                        for g, base, s in plan.schedules
                    ]
            if config.whiten_grad_off and airbench and step == whiten_steps:
                # The whitening bias is frozen (zero lr) from here: drop it from autograd,
                # which also removes the first conv's input gradient (one recompile, warmed up).
                model.whiten.bias.requires_grad_(False)
            if config.resize_in_model and freezable:
                model.train_size = size  # resized inside the (compiled) forward
            elif size is not None and size != inputs.size(-1):
                inputs = F.interpolate(
                    inputs, size=(size, size), mode="bilinear", align_corners=False, antialias=True
                )
            crop = scheduled(config.crop_schedule, step, total_steps, 0)
            if crop and crop < inputs.size(-1):
                # Window crops keep pixel scale (unlike downsampling); the final maps still end
                # on the 3x3 grid of 32 px input, so global-max statistics match test time.
                inputs = batch_crop(inputs, crop, crop_generator)
            if selector is not None:
                inputs, labels = selector.select(inputs, labels, step)
            plan.set_lr(step)
            if deepening:
                set_deepening(model, config, step, total_steps)
            if config.soft_pool_tau and airbench:
                model.soft_pool = size is not None and size != final_size
                if model.soft_pool:
                    tau = config.soft_pool_tau * (1 - step / max(1.0, switch_step))
                    model.pool_tau.fill_(max(tau, 1e-3))
            if config.compile_loss:
                loss = step_model(inputs, labels)
            elif config.mixup_alpha and step < config.mixup_until * total_steps:
                loss = mixup_loss(step_model, inputs, labels, config, stream.generator)
            elif isinstance(step_model, list):  # jointly trained ensemble members
                loss = sum(training_loss(m(inputs), labels, config) for m in step_model)
            else:
                outputs = step_model(inputs)
                if stream.scores is not None and not isinstance(outputs, tuple):
                    stream.observe(outputs, labels)
                if isinstance(outputs, tuple):  # multi-exit: main and stage-2 heads
                    main, aux = outputs
                    loss = training_loss(main, labels, config) + config.exit_weight * (
                        training_loss(aux, labels, config)
                    )
                elif targets.active:
                    soft, ls = targets.soft(labels, stream, step, total_steps)
                    loss = training_loss(outputs, labels, config, soft=soft, ls=ls)
                    targets.observe(outputs, labels, stream)
                else:
                    loss = training_loss(outputs, labels, config)
            loss.backward()
            sniffing = not switched and step >= switch_step - config.saliency_steps
            if saliency is not None and sniffing:
                saliency.observe(model)
            if masters is not None:
                masters.load_grads()
            plan.step()
            plan.zero_grad()
            step += 1
            if lookahead is not None and step % config.lookahead_every == 0:
                lookahead.update(decay=lookahead_decay(config, step, total_steps))
            if masters is not None:
                masters.sync()
            if step in snapshot_steps:
                snapshots.append(copy.deepcopy(model).eval())
            if step >= stop:
                break
    if freezable:
        for net in nets:
            net.frozen_groups = 0
            net.train_size = None
            net.soft_pool = False
            net.whiten.bias.requires_grad_(True)
            net.train()
            for module in net.modules():
                if isinstance(module, nn.Conv2d) and module is not net.whiten:
                    module.weight.requires_grad_(True)
    if deepening:
        set_deepening(model, config, total_steps, total_steps)
    if lookahead is not None:
        if config.lookahead_flush and step % config.lookahead_every:
            lookahead.update(decay=lookahead_decay(config, step, total_steps))
        # 1.0 returns the slow weights; lower values blend the final fast weights back in.
        lookahead.update(decay=config.lookahead_final_decay)
    if masters is not None:
        masters.sync()
    if config.bn_recal_batches:
        recalibrate_batchnorm(model, stream, config.bn_recal_batches)
    if config.head_refit_lambda and isinstance(model, AirbenchNet):
        refit_head(model, stream, config)
    return snapshots


def _narrow(model, narrow, plan, lookahead, saliency, config, stream, total_steps):
    """Transplant the wide net's highest-saliency channels into the narrow net and move the
    optimiser and lookahead state across; return the narrow net's training objects."""
    narrow_model, narrow_step = narrow
    keep = saliency.keep(model, narrowing.stage_widths(narrow_model), stream.generator)
    narrowing.transplant(model, narrow_model, keep)
    narrow_model.whiten.bias.requires_grad_(model.whiten.bias.requires_grad)
    narrow_model.train()
    new_plan = make_optimisation(narrow_model, config, total_steps, stream.steps_per_epoch)
    narrowing.transfer_momentum(plan.optimizers, new_plan.optimizers, model, narrow_model, keep)
    new_lookahead = None
    if lookahead is not None:
        new_lookahead = Lookahead(narrow_model)
        narrowing.transfer_slow_weights(lookahead.slow, model, new_lookahead.slow, keep)
    return narrow_model, narrow_step, new_plan, new_lookahead


class SoftTargets:
    """In-run soft targets (all state per trial, reset when built in ``fit``).

    * Online label smoothing (``ols_alpha``): the smoothing mass follows the model's own
      class confusions, S[y] = mean softmax over correctly classified class-y samples in the
      previous epoch.
    * Per-sample temporal targets (``pskd_alpha``, PS-KD style): blend in each image's softmax
      from its previous visit, ramped in after the resolution switch.
    * Label-smoothing annealing (``ls_end``): smoothing moves linearly to ``ls_end``.
    """

    def __init__(self, config: RecipeConfig, stream: TrainingStream, total_steps: int) -> None:
        self.config = config
        self.active = bool(config.ols_alpha or config.pskd_alpha or config.ls_end is not None)
        self.epoch = -1
        device = stream.labels.device
        if config.ols_alpha:
            self.confusion = torch.zeros(100, 100, device=device)
            self.smoothing: torch.Tensor | None = None
        if config.pskd_alpha:
            self.bank = torch.zeros(len(stream.labels), 100, device=device, dtype=torch.half)
            self.seen = torch.zeros(len(stream.labels), dtype=torch.bool, device=device)
            switch = max((f for f, size in config.res_schedule if size == 32), default=0.0)
            self.ramp_start = switch * total_steps

    def new_epoch(self) -> None:
        self.epoch += 1
        if self.config.ols_alpha and self.epoch > 0:
            rows = self.confusion.sum(1, keepdim=True)
            uniform = torch.full_like(self.confusion, 1 / 100)
            self.smoothing = torch.where(rows > 0, self.confusion / rows.clamp_min(1e-12), uniform)
            self.confusion.zero_()

    def soft(
        self, labels: torch.Tensor, stream: TrainingStream, step: int, total_steps: int
    ) -> tuple[torch.Tensor | None, float | None]:
        config = self.config
        ls = None
        if config.ls_end is not None:
            ls = config.label_smoothing + (config.ls_end - config.label_smoothing) * (
                step / total_steps
            )
        if not (config.ols_alpha or config.pskd_alpha):
            return None, ls
        eps = config.label_smoothing if ls is None else ls
        onehot = F.one_hot(labels, 100).float()
        target = onehot * (1 - eps) + eps / 100
        if config.ols_alpha and self.smoothing is not None:
            a = config.ols_alpha
            target = onehot * (1 - a) + a * self.smoothing[labels]
        if config.pskd_alpha and step >= self.ramp_start:
            frac = (step - self.ramp_start) / max(1.0, total_steps - self.ramp_start)
            alpha = config.pskd_alpha * frac
            idx = stream.last_idx
            prev = self.bank[idx].float()
            have = self.seen[idx].float()[:, None] * alpha
            target = target * (1 - have) + prev * have
        return target, ls

    @torch.no_grad()
    def observe(self, outputs: torch.Tensor, labels: torch.Tensor, stream: TrainingStream) -> None:
        probs = outputs.float().softmax(1)
        if self.config.ols_alpha:
            correct = (probs.argmax(1) == labels).float()[:, None]
            self.confusion.index_add_(0, labels, probs * correct)
        if self.config.pskd_alpha:
            idx = stream.last_idx
            self.bank[idx] = probs.half()
            self.seen[idx] = True


def weight_freeze_params(model: AirbenchNet, entry: tuple) -> list[torch.nn.Parameter]:
    """Conv weights selected by a ``weight_freeze`` entry (stage, start, end, which)."""
    stage, _, _, which = entry
    group = model.groups[stage]
    convs = (
        [group.conv1]
        if which == "conv1"
        else [m for m in group.modules() if isinstance(m, nn.Conv2d)]
    )
    return [m.weight for m in convs if m.weight.requires_grad]


def blended_size(
    config: RecipeConfig, step: int, switch_step: float, size: int | None
) -> int | None:
    """Stochastic resolution blend over ``res_blend_steps`` centred on the final switch: each
    step uses the final size with probability ramping 0 -> 1 (global RNG, seeded per trial)."""
    if not config.res_blend_steps or len(config.res_schedule) < 2:
        return size
    half = config.res_blend_steps / 2
    if not switch_step - half <= step < switch_step + half:
        return size
    p = (step - (switch_step - half) + 0.5) / config.res_blend_steps
    final = torch.rand(()).item() < p
    return config.res_schedule[-1][1] if final else config.res_schedule[-2][1]


def scale_momentum(plan: Optimisation, scale: float) -> None:
    """Partially reset SGD momentum (at the resolution switch)."""
    for optimizer in plan.optimizers:
        for state in optimizer.state.values():
            buffer = state.get("momentum_buffer")
            if buffer is not None:
                buffer.mul_(scale)


class TrainingObjective(nn.Module):
    """Model plus training loss, so ``compile_loss`` puts the loss inside the compiled graph."""

    def __init__(self, model: nn.Module, config: RecipeConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return training_loss(self.model(inputs), labels, self.config)


def lookahead_decay(config: RecipeConfig, step: int, total_steps: int) -> float:
    """airbench's lookahead pull, scaled to the update interval: base**every * progress**power."""
    base = config.lookahead_base**config.lookahead_every
    return base * (step / total_steps) ** config.lookahead_power


@torch.no_grad()
def refit_head(model: AirbenchNet, stream: TrainingStream, config: RecipeConfig) -> None:
    """Closed-form ridge refit of the linear head on full-resolution training features
    (timed, inside train): W = (F^T F + lambda n I)^-1 F^T Y with label-smoothed targets."""
    model.eval()
    images = stream.base
    pad = config.translate
    if pad:
        images = images[:, :, pad:-pad, pad:-pad]
    n = min(config.refit_samples, len(images))
    feats = torch.cat([model.features(chunk) for chunk in images[:n].split(2048)])
    labels = stream.labels[:n]
    eps = config.label_smoothing
    targets = torch.full((n, 100), eps / 100, device=feats.device)
    targets.scatter_(1, labels[:, None], 1 - eps + eps / 100)
    gram = feats.T @ feats + config.head_refit_lambda * n * torch.eye(
        feats.size(1), device=feats.device
    )
    weight = torch.linalg.solve(gram, feats.T @ targets).T  # [100, features]
    # Match the logit scale of the trained head so the refit only changes direction.
    current = model.head.weight.float()
    weight = weight * (current.norm() / weight.norm())
    model.head.weight.copy_(weight.to(model.head.weight.dtype))
    model.train()


# CIFAR-100 fine label -> superclass (the dataset's published 20-superclass taxonomy).
COARSE = (
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13,
    15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4,
    17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13,
    15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
)  # fmt: skip


def training_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    config: RecipeConfig,
    soft: torch.Tensor | None = None,
    ls: float | None = None,
) -> torch.Tensor:
    """Cross-entropy (default), PolyLoss-1 or squentropy, plus an optional superclass loss.

    ``soft`` replaces the smoothed one-hot target with an in-run soft target distribution;
    ``ls`` overrides the label smoothing (annealing)."""
    ls = config.label_smoothing if ls is None else ls
    if soft is not None:
        loss = -(soft * F.log_softmax(outputs.float(), dim=1)).sum()
    else:
        loss = F.cross_entropy(outputs, labels, label_smoothing=ls, reduction="sum")
    if config.loss == "poly1":
        p_true = outputs.softmax(-1).gather(1, labels[:, None]).squeeze(1)
        loss = loss + config.poly_eps * (1 - p_true).sum()
    elif config.loss == "squentropy":
        wrong = torch.ones_like(outputs).scatter_(1, labels[:, None], 0.0)
        loss = loss + ((outputs * wrong) ** 2).sum() / (outputs.size(1) - 1)
    if config.coarse_aux_weight:
        groups = torch.tensor(COARSE, device=outputs.device)
        coarse_logits = torch.full(
            (len(outputs), 20), float("-inf"), device=outputs.device
        ).scatter_reduce(1, groups.expand(len(outputs), -1), outputs, reduce="amax")
        # log-sum-exp within each superclass, computed stably around the per-group max
        shifted = (outputs - coarse_logits.gather(1, groups.expand(len(outputs), -1))).exp()
        sums = torch.zeros(len(outputs), 20, device=outputs.device).index_add_(1, groups, shifted)
        coarse = coarse_logits + sums.log()
        loss = loss + config.coarse_aux_weight * F.cross_entropy(
            coarse, groups[labels], reduction="sum"
        )
    return loss


def set_deepening(model: nn.Module, config: RecipeConfig, step: int, total_steps: int) -> None:
    """Progressive deepening: residual branches off until ``residual_start``, then ramped in."""
    start = config.residual_start * total_steps
    ramp = max(1.0, config.residual_ramp * total_steps)
    gate = min(1.0, max(0.0, (step - start) / ramp))
    for module in model.modules():
        if hasattr(module, "residual_active"):
            module.residual_active = step >= start
            module.residual_ramping = 0.0 < gate < 1.0
            module.residual_gate.fill_(gate)


def mixup_loss(
    step_model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    config: RecipeConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Mixup: blend each image with a permuted partner and mix the two label losses."""
    lam = float(
        torch.distributions.Beta(config.mixup_alpha, config.mixup_alpha).sample()
    )  # global RNG, seeded by the harness before every trial
    perm = torch.randperm(len(inputs), device=inputs.device, generator=generator)
    outputs = step_model(lam * inputs + (1 - lam) * inputs[perm])
    ls = config.label_smoothing
    return lam * F.cross_entropy(outputs, labels, label_smoothing=ls, reduction="sum") + (
        1 - lam
    ) * F.cross_entropy(outputs, labels[perm], label_smoothing=ls, reduction="sum")


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, stream: TrainingStream, batches: int) -> None:
    """Re-estimate BatchNorm running statistics for the final weights with a few no-grad
    training-mode passes over full-resolution training batches (timed, inside train)."""
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the recalibration batches
    model.train()
    for i, (inputs, _) in enumerate(stream.epoch(0)):
        if i == batches:
            break
        model(inputs)
    for m, momentum in zip(norms, momenta, strict=True):
        m.momentum = momentum
