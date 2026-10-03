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
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

from .config import RecipeConfig
from .data import TrainingStream
from .model import AirbenchNet, EnsembleNet, init_whitening, make_model, reset_model
from .train import TrainingObjective, fit


def build(context: BuildContext) -> SimpleNamespace:
    config = RecipeConfig.from_parameters(context.parameters)
    device = context.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if config.cudnn_benchmark_limit is not None:
            torch.backends.cudnn.benchmark_limit = config.cudnn_benchmark_limit
        # Set explicitly either way: Inductor config is process-global.
        torch._inductor.config.coordinate_descent_tuning = config.coordinate_descent
    if device.type != "cuda":
        # Compilation and fused SGD are GPU optimisations; CPU smoke tests run eagerly.
        config = replace(config, compile=False, fused_sgd=False)

    def compiled(net: nn.Module) -> nn.Module:
        if not config.compile:
            return net
        return torch.compile(net, mode=config.compile_mode, dynamic=False)

    if config.members > 1:
        # Jointly trained ensemble: separate nets, each compiled, trained on the same batches.
        nets = [make_model(config, device) for _ in range(config.members)]
        model: nn.Module = EnsembleNet(nets)
        step_model: nn.Module | list[nn.Module] = [compiled(net) for net in nets]
    else:
        model = make_model(config, device)
        step_model = compiled(TrainingObjective(model, config) if config.compile_loss else model)
    selector = make_model(config.selector_config(), device) if config.select_fraction < 1 else None
    narrow = None
    if config.switch_widths is not None:
        narrow_net = make_model(replace(config, widths=config.switch_widths), device)
        narrow = (narrow_net, compiled(narrow_net))
    state = SimpleNamespace(
        config=config,
        device=device,
        model=model,
        step_model=step_model,
        selector=selector,
        narrow=narrow,
    )
    if device.type == "cuda":
        _warm_up(state, context.eval_batch_size)
    return state


def prepare(state: SimpleNamespace, data: TrainingData, seed: int) -> None:
    _prepare(state, data, seed, state.config)


def train(state: SimpleNamespace) -> nn.Module:
    snapshots = fit(
        state.model, state.step_model, state.stream, state.config, state.selector, state.narrow
    )
    del state.stream
    if state.narrow is not None:
        return state.narrow[0]
    if snapshots:
        # Snapshot ensemble: copies of this trial's own weights, averaged at evaluation.
        return EnsembleNet([*snapshots, state.model])
    return state.model


def _prepare(
    state: SimpleNamespace, data: TrainingData, seed: int, config: RecipeConfig
) -> None:
    model, device = state.model, state.device
    nets = list(model.members) if isinstance(model, EnsembleNet) else [model]
    if state.selector is not None:
        nets.append(state.selector)
    for net in nets:
        reset_model(net)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    images = data.images.to(device, non_blocking=True).to(dtype).div_(255)
    labels = data.labels.to(device, non_blocking=True)
    std, mean = torch.std_mean(images, dim=(0, 2, 3))
    for net in nets:
        net.normalize.mean.copy_(mean.view(1, 3, 1, 1))
        net.normalize.std.copy_(std.view(1, 3, 1, 1))
        if isinstance(net, AirbenchNet):
            init_whitening(net, images[: config.whiten_samples])
    if config.head_mean_init and isinstance(model, AirbenchNet):
        init_head_from_class_means(model, images, labels, config)
    generator = torch.Generator(device=device).manual_seed(seed)
    state.stream = TrainingStream(images, labels, config, generator)


@torch.no_grad()
def init_head_from_class_means(
    model: AirbenchNet, images: torch.Tensor, labels: torch.Tensor, config: RecipeConfig
) -> None:
    """Data-dependent head init (timed, in prepare): row c = normalised mean pooled feature of
    class c over ``head_init_samples`` training images, scaled to the default init's norm."""
    n = min(config.head_init_samples, len(images))
    model.eval()  # freshly reset BN statistics: no running-stat updates from this pass
    feats = torch.cat([model.features(chunk) for chunk in images[:n].split(2048)])
    model.train()
    sums = torch.zeros(100, feats.size(1), device=feats.device).index_add_(0, labels[:n], feats)
    means = sums - sums.mean(0, keepdim=True)
    weight = model.head.weight
    rows = F.normalize(means, dim=1) * weight.float().norm(dim=1, keepdim=True).mean()
    if rows.size(0) == weight.size(0):
        weight.copy_(rows.to(weight.dtype))


def _warm_up(state: SimpleNamespace, eval_batch_size: int) -> None:
    """Run the full schedule on two batches of random data per epoch, then eval shapes.

    Every resolution and freezing phase of the real run therefore executes once outside the
    timer, so compilation, cuDNN autotuning and allocator growth never happen in a trial.
    ``prepare`` resets all state this touches.
    """
    config = state.config
    n = 2 * config.batch_size
    synthetic = TrainingData(
        images=torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
        labels=torch.randint(0, 100, (n,)),
    )
    _prepare(state, synthetic, seed=0, config=config)
    fit(state.model, state.step_model, state.stream, config, state.selector, state.narrow)
    del state.stream
    final = state.model if state.narrow is None else state.narrow[0]
    final.eval()
    with torch.inference_mode():
        for batch in (eval_batch_size, 10_000 % eval_batch_size or eval_batch_size):
            final(torch.rand(batch, 3, 32, 32, device=state.device))
    torch.cuda.synchronize(state.device)
