"""Mid-run channel narrowing (early-bird style).

A wide net trains through the cheap low-resolution phase; at the switch to the final
resolution, the highest-saliency channels of each stage are transplanted into a narrower net
(weights, BatchNorm statistics, momentum and lookahead state), which trains the expensive
full-resolution phase and is the returned model. A channel of stage g is shared by that stage's
conv outputs, its BatchNorms, the inputs of its conv2 and residual conv (residual add), and the
input of the next stage's conv1 or of the head, so one index set per stage selects it everywhere.
"""

from __future__ import annotations

import re

import torch
from torch import nn

_GROUP_TENSOR = re.compile(r"groups\.(\d)\.(.+)$")


def channel_dims(name: str) -> dict[int, int]:
    """Map each channel dimension of a state-dict tensor to the stage whose channels it holds."""
    match = _GROUP_TENSOR.match(name)
    if match:
        group, rest = int(match.group(1)), match.group(2)
        if rest.endswith("num_batches_tracked"):
            return {}
        if rest == "conv1.weight":
            return {0: group} | ({1: group - 1} if group > 0 else {})
        if rest in ("conv2.weight", "residual.0.weight"):
            return {0: group, 1: group}
        return {0: group}  # BatchNorm parameters and running statistics
    if name == "head.weight":
        return {1: 2}
    return {}


def sliced(name: str, tensor: torch.Tensor, keep: list[torch.Tensor]) -> torch.Tensor:
    for dim, group in channel_dims(name).items():
        tensor = tensor.index_select(dim, keep[group])
    return tensor


def stage_widths(model: nn.Module) -> list[int]:
    return [group.conv1.weight.size(0) for group in model.groups]


class Saliency:
    """Per-channel saliency of each stage: accumulated first-order Taylor importance
    (sum over a channel's parameters of w * dL/dw, squared per step), weight norm, or random."""

    def __init__(self, model: nn.Module, mode: str) -> None:
        self.mode = mode
        device = model.head.weight.device
        self.scores = [torch.zeros(w, device=device) for w in stage_widths(model)]

    @torch.no_grad()
    def observe(self, model: nn.Module) -> None:
        if self.mode != "taylor":
            return
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            product = p.float() * p.grad.float()
            for dim, group in channel_dims(name).items():
                others = [d for d in range(p.ndim) if d != dim]
                contribution = product.sum(others) if others else product
                self.scores[group] += contribution.square()

    @torch.no_grad()
    def keep(
        self, model: nn.Module, widths: list[int], generator: torch.Generator
    ) -> list[torch.Tensor]:
        scores = self.scores
        if self.mode == "norm":
            scores = [torch.zeros_like(s) for s in scores]
            for name, p in model.named_parameters():
                for dim, group in channel_dims(name).items():
                    others = [d for d in range(p.ndim) if d != dim]
                    squared = p.float().square()
                    scores[group] += squared.sum(others) if others else squared
        elif self.mode == "random":
            scores = [
                torch.rand(len(s), device=s.device, generator=generator) for s in self.scores
            ]
        return [torch.topk(s, w).indices.sort().values for s, w in zip(scores, widths, strict=True)]


@torch.no_grad()
def transplant(wide: nn.Module, narrow: nn.Module, keep: list[torch.Tensor]) -> None:
    """Copy every state-dict tensor of the wide net, restricted to the kept channels."""
    target = narrow.state_dict()
    for name, value in wide.state_dict().items():
        target[name].copy_(sliced(name, value, keep))


def transfer_momentum(old_optimizers, new_optimizers, wide, narrow, keep) -> None:
    """Seed the narrow net's SGD momentum buffers with the kept slices of the wide net's."""
    names = {id(p): name for name, p in wide.named_parameters()}
    targets = dict(narrow.named_parameters())
    for old, new in zip(old_optimizers, new_optimizers, strict=True):
        for p, state in old.state.items():
            buffer = state.get("momentum_buffer")
            if buffer is not None:
                name, target = names[id(p)], targets[names[id(p)]]
                # Match the parameter's memory format (channels_last): fused SGD requires it.
                state_buffer = torch.empty_like(target).copy_(sliced(name, buffer, keep))
                new.state[target]["momentum_buffer"] = state_buffer


@torch.no_grad()
def transfer_slow_weights(
    slow: list[torch.Tensor], wide: nn.Module, new_slow: list[torch.Tensor], keep
) -> None:
    """Restrict lookahead slow weights (ordered as the floating state-dict entries)."""
    floating = (torch.half, torch.float)
    names = [k for k, v in wide.state_dict().items() if v.dtype in floating]
    for name, old, new in zip(names, slow, new_slow, strict=True):
        new.copy_(sliced(name, old, keep))
