"""Harness adapter for the team_segal CIFAR-100 recipe.

``build`` constructs the network and warms it up on synthetic data only. ``prepare`` resets
every learned or data-derived tensor in place and does all work that touches real training
data (transfer, scaling, input statistics, whitening). ``train`` runs the optimisation and
returns the eager model; the compiled wrapper is used for training steps only, so evaluation
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
from .model import init_whitening, make_model, reset_model
from .train import fit


def build(context: BuildContext) -> SimpleNamespace:
    config = RecipeConfig.from_parameters(context.parameters)
    device = context.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        # Compilation and fused SGD are GPU optimisations; CPU smoke tests run eagerly.
        config = replace(config, compile=False, fused_sgd=False)
    model = make_model(config, device)
    # One specialised graph per training resolution (dynamic=False avoids a slower
    # dynamic-shape recompile when the resolution changes).
    step_model = (
        torch.compile(model, mode=config.compile_mode, dynamic=False) if config.compile else model
    )
    state = SimpleNamespace(config=config, device=device, model=model, step_model=step_model)
    if device.type == "cuda":
        _warm_up(state, context.eval_batch_size)
    return state


def prepare(state: SimpleNamespace, data: TrainingData, seed: int) -> None:
    model, device, config = state.model, state.device, state.config
    reset_model(model)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    images = data.images.to(device, non_blocking=True).to(dtype).div_(255)
    labels = data.labels.to(device, non_blocking=True)
    std, mean = torch.std_mean(images, dim=(0, 2, 3))
    model.normalize.mean.copy_(mean.view(1, 3, 1, 1))
    model.normalize.std.copy_(std.view(1, 3, 1, 1))
    init_whitening(model, images[: config.whiten_samples])
    generator = torch.Generator(device=device).manual_seed(seed)
    state.stream = TrainingStream(images, labels, config, generator)


def train(state: SimpleNamespace) -> nn.Module:
    fit(state.model, state.step_model, state.stream, state.config)
    del state.stream
    return state.model


def _warm_up(state: SimpleNamespace, eval_batch_size: int) -> None:
    """Run the full schedule on two batches of random data per epoch, then the eval shapes.

    Every resolution phase of the real run executes once outside the timer, so compilation,
    cuDNN autotuning and allocator growth never happen in a trial. ``prepare`` resets all
    state this touches.
    """
    n = 2 * state.config.batch_size
    synthetic = TrainingData(
        images=torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
        labels=torch.randint(0, 100, (n,)),
    )
    prepare(state, synthetic, seed=0)
    train(state)
    state.model.eval()
    with torch.inference_mode():
        for batch in (eval_batch_size, 10_000 % eval_batch_size or eval_batch_size):
            state.model(torch.rand(batch, 3, 32, 32, device=state.device))
    torch.cuda.synchronize(state.device)
