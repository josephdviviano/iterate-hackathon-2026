"""Declarative recipe parameters: one typed, validated source of every tunable choice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

Arch = Literal["airbench", "resnet9", "convmixer"]
Flip = Literal["alternating", "random", "none"]
Optimizer = Literal["sgd", "muon"]


@dataclass(frozen=True)
class RecipeConfig:
    """Hyperparameters of one recipe; defaults are airbench94 adapted to 100 classes.

    Optimiser hyperparameters use airbench's decoupled form: ``lr`` and ``weight_decay`` are
    per 1024 examples and independent of ``momentum``, so each can be tuned on its own.
    """

    arch: Arch = "airbench"
    widths: tuple[int, int, int] = (64, 256, 256)
    width_mult: float = 1.0
    block_depth: Literal[2, 3] = 2
    # Per-stage overrides: depth (2 or 3) and whether to max-pool before the widening conv.
    stage_depths: tuple[int, int, int] | tuple[()] = ()
    pool_first: tuple[bool, bool, bool] | tuple[()] = ()
    epochs: float = 10.0
    batch_size: int = 1024
    lr: float = 11.5
    momentum: float = 0.85
    weight_decay: float = 0.0153
    bias_scaler: float = 64.0
    label_smoothing: float = 0.2
    lr_start: float = 0.2
    lr_peak_frac: float = 0.23
    lr_end: float = 0.07
    lookahead: bool = True
    whiten_bias_epochs: float = 3.0
    whiten_samples: int = 5000
    bn_momentum: float = 0.6
    scaling_factor: float = 1 / 9
    flip: Flip = "alternating"
    translate: int = 2
    cutout: int = 0
    compile: bool = False
    compile_mode: str = "default"
    # Pooling implementation: "torch" (indexed max-pool) or "amax" (reshape + amax).
    pool_impl: str = "torch"
    fused_sgd: bool = False
    # Muon mode (airbench94_muon): Muon on conv filters, SGD on biases and the head, linear
    # decay to zero, whitening-bias lr decaying over ``whiten_bias_epochs``.
    optimizer: Optimizer = "sgd"
    muon_lr: float = 0.24
    muon_momentum: float = 0.6
    muon_ns_steps: int = 3
    muon_bias_lr: float = 0.053
    muon_head_lr: float = 0.67
    muon_weight_decay: float = 2e-6
    # Logits divided by the head's fan-in, head initialised at unit std (airbench94_muon).
    head_norm: bool = False
    # Progressive resizing: ((start_fraction, size), ...); empty trains at full resolution.
    res_schedule: tuple[tuple[float, int], ...] = ()
    # In-run example selection (airbench96_faster): a small selector net scores each batch and
    # the main net trains only on the highest-loss ``select_fraction`` of it. The selector
    # trains on its own selected losses every ``selector_update_every`` steps.
    select_fraction: float = 1.0
    selector_widths: tuple[int, int, int] = (32, 64, 64)
    selector_update_every: int = 4
    # Progressive freezing: ((start_fraction, frozen_stages), ...) for airbench nets.
    freeze_schedule: tuple[tuple[float, int], ...] = ()
    # Coverage levers (CIFAR-10 lineage, competitor and novel alternatives).
    activation: str = "gelu"
    whiten_kernel: int = 2
    head_pool: str = "max"
    brightness: float = 0.0
    contrast: float = 0.0
    mixup_alpha: float = 0.0
    mixup_until: float = 0.5
    lookahead_flush: bool = False
    bn_recal_batches: int = 0
    jitter_mode: str = "multiplicative"
    global_pool: str = ""
    resize_in_model: bool = False
    muon_coefficients: str = "airbench"
    muon_renorm: str = "every"
    muon_decoupled_wd: float = 0.0
    # Representation alternatives (X-004).
    rep_branch: bool = False
    residual_start: float = 0.0
    residual_ramp: float = 0.1
    loss: str = "ce"
    poly_eps: float = 1.0
    coarse_aux_weight: float = 0.0
    convmixer_dim: int = 512
    convmixer_depth: int = 8
    convmixer_kernel: int = 5
    convmixer_patch: int = 2
    # Structural exploration (X-006).
    members: int = 1
    snapshot_fracs: tuple[float, ...] = ()
    exit_weight: float = 0.0
    exit_eval_weight: float = 0.0
    head_refit_lambda: float = 0.0
    refit_samples: int = 20000
    lr_shape: str = "triangle"
    lr_decay_start: float = 0.6
    lookahead_every: int = 5
    lookahead_power: float = 3.0
    order: str = "random"
    # Agent round 1 (representation and rule-legal structure).
    ols_alpha: float = 0.0
    pskd_alpha: float = 0.0
    ls_end: float | None = None
    head_expand: int = 0
    pool_overlap: bool = False
    stage3_ceil: bool = False
    cosine_head_scale: float = 0.0
    cosine_subcenters: int = 1
    head_mean_init: bool = False
    head_init_samples: int = 5000
    # Agent round 1 (optimisation dynamics).
    master_fp32: bool = False
    clean_tail_epochs: float | None = None
    head_lr_mult: float = 1.0
    head_wd_mult: float = 1.0
    conv_wd_mult: float = 1.0
    init_gain: float = 1.0
    switch_momentum_scale: float = 1.0
    res_blend_steps: int = 0
    etf_head: bool = False
    soft_pool_tau: float = 0.0
    # Agent round 1 (exact systems levers).
    skip_discarded: bool = False
    whiten_grad_off: bool = False
    cudnn_benchmark_limit: int | None = None
    coordinate_descent: bool = False
    compile_loss: bool = False
    # Zero-overhead data pruning from per-image scores banked during the normal forward pass.
    prune_frac: float = 0.0
    prune_start: int = 1
    prune_mode: str = "easy"
    prune_score: str = "loss"
    prune_alpha: float = 1.0
    # Mid-run channel narrowing at the final resolution switch (see narrowing.py).
    switch_widths: tuple[int, int, int] | None = None
    narrow_saliency: str = "taylor"
    saliency_steps: int = 24
    # Agent round 2 (budget reallocation).
    stage1_cooldown: tuple[float, float] | None = None
    bias_scaler_final: float | None = None
    # Teammates' hypothesis-branch block: activation after the residual add.
    post_add_activation: bool = False
    # Structured (unlearned) initialisation of the non-identity conv rows, and SkipInit gates.
    conv_init: str = "kaiming"
    residual_gate_init: float | None = None

    def _validate_narrowing(self) -> None:
        widths = self.switch_widths
        if len(widths) != 3 or any(not 8 <= n <= w or n % 8 for n, w in zip(widths, self.widths)):
            raise ValueError("switch_widths must be three multiples of 8, each <= widths")
        if self.narrow_saliency not in ("taylor", "norm", "random") or self.saliency_steps < 1:
            raise ValueError("narrow_saliency must be taylor/norm/random, saliency_steps >= 1")
        unsupported = {
            "arch": self.arch != "airbench",
            "res_schedule": len(self.res_schedule) < 2,
            "width_mult": self.width_mult != 1,
            "members": self.members > 1,
            "rep_branch": self.rep_branch,
            "master_fp32": self.master_fp32,
            "snapshot_fracs": bool(self.snapshot_fracs),
            "exit heads": bool(self.exit_weight or self.exit_eval_weight),
            "head_expand": bool(self.head_expand),
            "cosine_head_scale": bool(self.cosine_head_scale),
            "etf_head": self.etf_head,
            "res_blend_steps": bool(self.res_blend_steps),
            "residual_start": bool(self.residual_start),
            "select_fraction": self.select_fraction < 1,
            "residual_gate_init": self.residual_gate_init is not None,
        }
        clashes = sorted(k for k, v in unsupported.items() if v)
        if clashes:
            raise ValueError(f"switch_widths is not supported with {clashes}")

    def selector_config(self) -> RecipeConfig:
        """The small airbench94-shaped selector used for in-run example selection."""
        return RecipeConfig(
            widths=self.selector_widths, epochs=self.epochs, batch_size=self.batch_size
        )

    @property
    def scaled_widths(self) -> tuple[int, int, int]:
        """Block widths after the multiplier, rounded to a multiple of 8 for tensor cores."""
        a, b, c = (max(8, int(round(w * self.width_mult / 8)) * 8) for w in self.widths)
        return a, b, c

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> RecipeConfig:
        """Build a config from harness JSON parameters, failing loudly on unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise ValueError(f"Unknown recipe parameters: {unknown}; known: {sorted(known)}")
        values = dict(parameters)
        if "widths" in values:
            widths = values["widths"]
            if len(widths) != 3 or not all(isinstance(w, int) and w > 0 for w in widths):
                raise ValueError("widths must be three positive integers")
            values["widths"] = tuple(widths)
        for key in ("stage_depths", "pool_first"):
            if key in values:
                values[key] = tuple(values[key])
        if "selector_widths" in values:
            values["selector_widths"] = tuple(values["selector_widths"])
        if values.get("stage1_cooldown") is not None:
            values["stage1_cooldown"] = tuple(values["stage1_cooldown"])
        if values.get("switch_widths") is not None:
            values["switch_widths"] = tuple(values["switch_widths"])
        if "snapshot_fracs" in values:
            values["snapshot_fracs"] = tuple(values["snapshot_fracs"])
        for key in ("res_schedule", "freeze_schedule"):
            if key in values:
                values[key] = tuple(tuple(entry) for entry in values[key])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.arch not in ("airbench", "resnet9", "convmixer"):
            raise ValueError(f"arch must be airbench, resnet9 or convmixer, not {self.arch!r}")
        if self.members < 1 or self.lookahead_every < 1 or self.refit_samples < 1:
            raise ValueError("members, lookahead_every and refit_samples must be >= 1")
        if any(not 0 < f < 1 for f in self.snapshot_fracs):
            raise ValueError("snapshot_fracs must lie in (0, 1)")
        if self.stage1_cooldown is not None and (
            len(self.stage1_cooldown) != 2
            or not 0 <= self.stage1_cooldown[0] < self.stage1_cooldown[1] <= 1
            or self.optimizer != "sgd"
        ):
            raise ValueError("stage1_cooldown must be (start, end) with 0 <= start < end <= 1")
        if self.conv_init not in ("kaiming", "orthogonal", "dct", "zero"):
            raise ValueError("conv_init must be kaiming, orthogonal, dct or zero")
        if self.bias_scaler_final is not None and self.bias_scaler_final <= 0:
            raise ValueError("bias_scaler_final must be positive")
        if self.lr_shape not in ("triangle", "wsd", "wsd_sqrt", "cosine") or self.order not in (
            "random",
            "balanced",
        ):
            raise ValueError("lr_shape must be triangle/wsd/cosine and order random/balanced")
        if min(self.ols_alpha, self.pskd_alpha, self.head_expand, self.cosine_head_scale) < 0:
            raise ValueError("ols_alpha, pskd_alpha, head_expand, cosine_head_scale must be >= 0")
        if min(self.head_lr_mult, self.init_gain) <= 0:
            raise ValueError("head_lr_mult and init_gain must be positive")
        scales = (self.head_wd_mult, self.conv_wd_mult, self.switch_momentum_scale)
        if min(*scales, self.soft_pool_tau) < 0:
            raise ValueError("wd multipliers, switch_momentum_scale, soft_pool_tau must be >= 0")
        if self.res_blend_steps < 0 or (
            self.clean_tail_epochs is not None and self.clean_tail_epochs < 0
        ):
            raise ValueError("res_blend_steps and clean_tail_epochs must be non-negative")
        soft = self.ols_alpha or self.pskd_alpha or self.ls_end is not None
        if self.compile_loss and (self.members > 1 or self.exit_weight or self.mixup_alpha or soft):
            raise ValueError("compile_loss supports only the single-net cross-entropy path")
        if not 0 <= self.prune_frac < 1 or self.prune_start < 1 or self.prune_alpha < 0:
            raise ValueError("prune_frac must be in [0, 1), prune_start >= 1, prune_alpha >= 0")
        if self.prune_mode not in ("easy", "hard", "split", "soft") or self.prune_score not in (
            "loss",
            "grad",
        ):
            raise ValueError("prune_mode must be easy/hard/split/soft and prune_score loss/grad")
        if self.prune_frac and (self.compile_loss or self.members > 1 or self.mixup_alpha):
            raise ValueError("prune_frac needs per-image logits (single net, no mixup)")
        if self.switch_widths is not None:
            self._validate_narrowing()
        if self.cosine_subcenters < 1 or self.head_init_samples < 1:
            raise ValueError("cosine_subcenters and head_init_samples must be >= 1")
        if min(self.exit_weight, self.exit_eval_weight, self.head_refit_lambda) < 0:
            raise ValueError("exit weights and head_refit_lambda must be non-negative")
        if self.loss not in ("ce", "poly1", "squentropy"):
            raise ValueError("loss must be ce, poly1 or squentropy")
        if not 0 <= self.residual_start < 1 or not 0 < self.residual_ramp <= 1:
            raise ValueError("residual_start must be in [0, 1) and residual_ramp in (0, 1]")
        if self.coarse_aux_weight < 0:
            raise ValueError("coarse_aux_weight must be non-negative")
        if self.block_depth not in (2, 3):
            raise ValueError("block_depth must be 2 or 3")
        if self.stage_depths and (
            len(self.stage_depths) != 3 or any(d not in (2, 3) for d in self.stage_depths)
        ):
            raise ValueError("stage_depths must be three values of 2 or 3")
        if self.pool_first and (
            len(self.pool_first) != 3 or any(type(p) is not bool for p in self.pool_first)
        ):
            raise ValueError("pool_first must be three booleans")
        if self.flip not in ("alternating", "random", "none"):
            raise ValueError(f"flip must be alternating, random or none, not {self.flip!r}")
        positive = {
            "width_mult": self.width_mult,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "whiten_samples": self.whiten_samples,
            "scaling_factor": self.scaling_factor,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.momentum < 1 or not 0 < self.bn_momentum < 1:
            raise ValueError("momentum must be in [0, 1) and bn_momentum in (0, 1)")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0 < self.lr_peak_frac < 1:
            raise ValueError("lr_peak_frac must be in (0, 1)")
        if self.translate < 0 or self.cutout < 0 or self.weight_decay < 0:
            raise ValueError("translate, cutout and weight_decay must be non-negative")
        if self.optimizer not in ("sgd", "muon"):
            raise ValueError(f"optimizer must be sgd or muon, not {self.optimizer!r}")
        if not (self.muon_lr > 0 and self.muon_bias_lr > 0 and self.muon_head_lr > 0):
            raise ValueError("Muon learning rates must be positive")
        if not 0 <= self.muon_momentum < 1 or self.muon_ns_steps < 1:
            raise ValueError("muon_momentum must be in [0, 1) and muon_ns_steps at least 1")
        starts = [entry[0] for entry in self.res_schedule]
        if any(len(entry) != 2 for entry in self.res_schedule) or starts != sorted(starts):
            raise ValueError("res_schedule must be ascending (start_fraction, size) pairs")
        if any(not 0 <= f < 1 or not 8 <= size <= 32 for f, size in self.res_schedule):
            raise ValueError("res_schedule fractions must be in [0, 1) and sizes in [8, 32]")
        freeze_starts = [entry[0] for entry in self.freeze_schedule]
        if freeze_starts != sorted(freeze_starts) or any(
            len(e) != 2 or not 0 <= e[0] < 1 or e[1] not in (0, 1, 2) for e in self.freeze_schedule
        ):
            raise ValueError("freeze_schedule must be ascending (start_fraction, 0|1|2) pairs")
        if not 0 < self.select_fraction <= 1 or self.selector_update_every < 1:
            raise ValueError("select_fraction must be in (0, 1] and selector_update_every >= 1")
        if self.pool_impl not in ("torch", "amax"):
            raise ValueError("pool_impl must be torch or amax")
        if self.activation not in ("gelu", "silu", "celu"):
            raise ValueError("activation must be gelu, silu or celu")
        if self.whiten_kernel not in (2, 3) or self.head_pool not in ("max", "maxmean"):
            raise ValueError("whiten_kernel must be 2 or 3 and head_pool max or maxmean")
        if min(self.brightness, self.contrast, self.mixup_alpha, self.bn_recal_batches) < 0:
            raise ValueError("jitter, mixup_alpha and bn_recal_batches must be non-negative")
        if self.jitter_mode not in ("multiplicative", "hiverge"):
            raise ValueError("jitter_mode must be multiplicative or hiverge")
        if self.global_pool not in ("", "torch", "flatmax", "amax"):
            raise ValueError("global_pool must be torch, flatmax or amax")
        if self.muon_coefficients not in ("airbench", "hiverge"):
            raise ValueError("muon_coefficients must be airbench or hiverge")
        if self.muon_renorm not in ("every", "hiverge") or self.muon_decoupled_wd < 0:
            raise ValueError("muon_renorm must be every or hiverge; muon_decoupled_wd >= 0")
        if not 0 <= self.mixup_until <= 1:
            raise ValueError("mixup_until must be in [0, 1]")
        modes = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
        if self.compile_mode not in modes:
            raise ValueError(f"compile_mode must be one of {modes}")
