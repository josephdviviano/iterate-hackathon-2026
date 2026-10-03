"""Training FLOPs per epoch for recipe parameters (secondary time proxy, decision D-001).

    python research/flops.py '{"width_mult": 2.0}' '{"widths": [128, 384, 512], "block_depth": 3}'

Counts multiply-accumulates of conv and linear layers in one forward pass at the training
resolution, and reports 3x forward FLOPs (forward plus backward) times 50,000 images. Lower
resolutions in ``res_schedule`` are weighted by their share of training steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

from benchmark.worker import load_submission

load_submission(Path(__file__).resolve().parents[1] / "research" / "lab_recipe")
from benchmark._submission.config import RecipeConfig  # noqa: E402
from benchmark._submission.model import make_model  # noqa: E402

TRAIN_IMAGES = 50_000


def forward_macs(model: nn.Module, size: int) -> int:
    total = 0

    def count(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            total += output.numel() * k // module.groups
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    hooks = [m.register_forward_hook(count) for m in model.modules()]
    with torch.no_grad():
        model.eval()(torch.rand(1, 3, size, size))
    for hook in hooks:
        hook.remove()
    return total


def epoch_tflops(parameters: dict) -> float:
    config = RecipeConfig.from_parameters(parameters)
    model = make_model(config, torch.device("cpu"))
    segments = list(config.res_schedule) or [(0.0, 32)]
    bounds = [start for start, _ in segments[1:]] + [1.0]
    macs = sum(
        (end - start) * forward_macs(model, size)
        for (start, size), end in zip(segments, bounds, strict=True)
    )
    if segments[0][0] > 0:
        macs += segments[0][0] * forward_macs(model, 32)
    return 3 * 2 * macs * TRAIN_IMAGES / 1e12


def main() -> None:
    for text in sys.argv[1:]:
        parameters = json.loads(text)
        print(f"{epoch_tflops(parameters):8.2f} TFLOP/epoch  {json.dumps(parameters)}")


if __name__ == "__main__":
    main()
