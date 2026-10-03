"""Recipe parameters: one typed, validated source of every tunable choice."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class RecipeConfig:
    """Hyperparameters of the recipe; the defaults are the submitted recipe.

    Optimiser hyperparameters use airbench's decoupled form: ``lr`` and ``weight_decay`` are
    per 1024 examples and independent of ``momentum``, so each can be tuned on its own.
    """

    widths: tuple[int, int, int] = (128, 384, 640)
    epochs: float = 8.25
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
    scaling_factor: float = 1 / 6
    translate: int = 2
    # Progressive resizing: ((start_fraction, size), ...), ending at the 32 px test resolution.
    res_schedule: tuple[tuple[float, int], ...] = ((0.0, 20), (0.5, 32))
    compile: bool = True
    # max-autotune adds Inductor autotuning and CUDA graphs (about 2 min of untimed build).
    compile_mode: str = "max-autotune"
    fused_sgd: bool = True

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> RecipeConfig:
        """Build a config from harness JSON parameters, failing loudly on unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise ValueError(f"Unknown recipe parameters: {unknown}; known: {sorted(known)}")
        values = dict(parameters)
        if "widths" in values:
            values["widths"] = tuple(values["widths"])
        if "res_schedule" in values:
            values["res_schedule"] = tuple(tuple(entry) for entry in values["res_schedule"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.widths) != 3 or not all(isinstance(w, int) and w > 0 for w in self.widths):
            raise ValueError("widths must be three positive integers")
        positive = {
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
        if self.translate < 0 or self.weight_decay < 0:
            raise ValueError("translate and weight_decay must be non-negative")
        modes = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
        if self.compile_mode not in modes:
            raise ValueError(f"compile_mode must be one of {modes}")
        starts = [entry[0] for entry in self.res_schedule]
        if any(len(entry) != 2 for entry in self.res_schedule) or starts != sorted(starts):
            raise ValueError("res_schedule must be ascending (start_fraction, size) pairs")
        if any(not 0 <= f < 1 or not 8 <= size <= 32 for f, size in self.res_schedule):
            raise ValueError("res_schedule fractions must be in [0, 1) and sizes in [8, 32]")
