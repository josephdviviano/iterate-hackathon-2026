"""Adaptive conformal inference over binary claims (Gibbs and Candès, 2021).

Each step supplies the nonconformity score of every label and the true label.
The set at step t holds the labels whose score is at most the current quantile;
the level moves by gamma towards the target after every miss or hit, so the
long-run coverage is 1 - alpha whatever the sequence. A set with one label is
a commitment; the share of committed steps is reported beside coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AciResult:
    coverage: float
    committed: float
    sizes: tuple[int, ...]
    covered: tuple[bool, ...]

    @property
    def n(self) -> int:
        return len(self.sizes)


def aci(scores: Sequence[Sequence[float]], truth: Sequence[int], *, alpha: float = 0.1, gamma: float = 0.05) -> AciResult:
    """``scores[t][k]`` is the nonconformity of label k at step t; ``truth[t]`` is the true label index."""
    level = alpha
    history: list[float] = []
    sizes: list[int] = []
    covered: list[bool] = []
    for step, true in zip(scores, truth):
        if history:
            q = float(np.quantile(history, min(1.0, max(0.0, 1.0 - level))))
        else:
            q = float("inf")
        members = [k for k, s in enumerate(step) if s <= q]
        hit = true in members
        sizes.append(len(members))
        covered.append(hit)
        level += gamma * (alpha - (0.0 if hit else 1.0))
        history.append(float(step[true]))
    if not sizes:
        return AciResult(float("nan"), float("nan"), (), ())
    return AciResult(float(np.mean(covered)), float(np.mean([s == 1 for s in sizes])), tuple(sizes), tuple(covered))


def binary_scores(p: Sequence[float]) -> list[list[float]]:
    """Nonconformity for labels (0, 1) from the probability of label 1."""
    return [[float(v), 1.0 - float(v)] for v in p]
