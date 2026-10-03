"""Declarative recipe parameters: one typed, validated source of every tunable choice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

Arch = Literal["airbench", "resnet9"]
Flip = Literal["alternating", "random", "none"]


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
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.arch not in ("airbench", "resnet9"):
            raise ValueError(f"arch must be airbench or resnet9, not {self.arch!r}")
        if self.block_depth not in (2, 3):
            raise ValueError("block_depth must be 2 or 3")
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
