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
        for key in ("res_schedule", "freeze_schedule"):
            if key in values:
                values[key] = tuple(tuple(entry) for entry in values[key])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.arch not in ("airbench", "resnet9", "convmixer"):
            raise ValueError(f"arch must be airbench, resnet9 or convmixer, not {self.arch!r}")
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
