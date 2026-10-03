"""Harness adapter for the parametrised airbench-lineage CIFAR-100 recipe.

``build`` constructs the network and warms it up on synthetic data only. ``prepare`` resets
every learned or data-derived tensor in place and does all work that touches real training
data (transfer, scaling, input statistics, whitening). ``train`` runs the optimisation and
returns the eager model; any compiled wrapper is used for training steps only, so evaluation
never triggers compilation.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from benchmark.api import BuildContext, TrainingData

from .config import RecipeConfig
from .data import TrainingStream
from .model import AirbenchNet, init_whitening, make_model, reset_model
from .train import fit


def build(context: BuildContext) -> SimpleNamespace:
    config = RecipeConfig.from_parameters(context.parameters)
    device = context.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = make_model(config, device)
    step_model = torch.compile(model) if config.compile else model
    state = SimpleNamespace(config=config, device=device, model=model, step_model=step_model)
    if device.type == "cuda":
        _warm_up(state, context.eval_batch_size)
    return state


def prepare(state: SimpleNamespace, data: TrainingData, seed: int) -> None:
    _prepare(state, data, seed, state.config)


def train(state: SimpleNamespace) -> nn.Module:
    fit(state.model, state.step_model, state.stream, state.config)
    del state.stream
    return state.model


def _prepare(
    state: SimpleNamespace, data: TrainingData, seed: int, config: RecipeConfig
) -> None:
    model, device = state.model, state.device
    reset_model(model)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    images = data.images.to(device, non_blocking=True).to(dtype).div_(255)
    labels = data.labels.to(device, non_blocking=True)
    pixels = images.float()
    model.normalize.mean.copy_(pixels.mean(dim=(0, 2, 3)).view(1, 3, 1, 1))
    model.normalize.std.copy_(pixels.std(dim=(0, 2, 3)).view(1, 3, 1, 1))
    del pixels
    if isinstance(model, AirbenchNet):
        init_whitening(model, images[: config.whiten_samples])
        model.whiten.bias.requires_grad = True
    generator = torch.Generator(device=device).manual_seed(seed)
    state.stream = TrainingStream(images, labels, config, generator)


def _warm_up(state: SimpleNamespace, eval_batch_size: int) -> None:
    """Exercise training (both whitening-bias states) and eval shapes on random data.

    This compiles and autotunes outside the timer. ``prepare`` resets all state it touches.
    """
    config = state.config
    n = 2 * config.batch_size
    synthetic = TrainingData(
        images=torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
        labels=torch.randint(0, 100, (n,)),
    )
    warm = replace(config, epochs=2.0, whiten_bias_epochs=1.0)
    _prepare(state, synthetic, seed=0, config=warm)
    fit(state.model, state.step_model, state.stream, warm)
    del state.stream
    state.model.eval()
    with torch.inference_mode():
        for batch in (eval_batch_size, 10_000 % eval_batch_size or eval_batch_size):
            state.model(torch.rand(batch, 3, 32, 32, device=state.device))
    torch.cuda.synchronize(state.device)
