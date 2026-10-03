"""Random-restart local search over k-atom measures with a float objective.

The naive baseline, equivalent to a search with no structural knowledge.
The best float candidate is rounded to rationals, repaired and certified; the
reported value is the certified one.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .problem import Instance
from .verify import Certificate, Rejected, certify, count_vectors


class FloatObjective:
    def __init__(self, inst: Instance, k: int):
        self.inst, self.k = inst, k
        cv = count_vectors(inst.n, k)
        self.C = np.array([c for c, _ in cv], dtype=float)
        self.coef = np.array([coef for _, coef in cv], dtype=float)
        self.m, self.t = float(inst.m), float(inst.t)

    def weights_for(self, atoms: np.ndarray, free: np.ndarray) -> np.ndarray | None:
        """free: k-2 weights; the last two solve the two linear constraints."""
        W, M = free.sum(), float(free @ atoms[:-2])
        a1, a2 = atoms[-2], atoms[-1]
        if abs(a2 - a1) < 1e-12:
            return None
        w2 = ((self.m - M) - a1 * (1 - W)) / (a2 - a1)
        w1 = (1 - W) - w2
        w = np.concatenate([free, [w1, w2]])
        return w if (w >= 0).all() else None

    def value(self, atoms: np.ndarray, w: np.ndarray) -> float:
        mask = self.C @ atoms <= self.t + 1e-12
        with np.errstate(divide="ignore"):
            logw = np.log(np.maximum(w, 1e-300))
        return float((self.coef[mask] * np.exp(self.C[mask] @ logw)).sum())


def local_search(inst: Instance, k: int, restarts: int = 20, steps: int = 400, seed: int = 0) -> Certificate | None:
    rng = np.random.default_rng(seed)
    obj = FloatObjective(inst, k)
    best = (-1.0, None, None)
    for _ in range(restarts):
        atoms = np.sort(rng.uniform(0, 1, k))
        free = rng.dirichlet(np.ones(k)) [: k - 2] if k > 2 else np.zeros(0)
        w = obj.weights_for(atoms, free)
        if w is None:
            continue
        cur = obj.value(atoms, w)
        step = 0.2
        for s in range(steps):
            na = np.clip(atoms + rng.normal(0, step, k), 0, 1)
            nf = np.clip(free + rng.normal(0, step, free.shape), 0, 1)
            nw = obj.weights_for(na, nf)
            if nw is None:
                continue
            v = obj.value(na, nw)
            if v > cur:
                atoms, free, w, cur = na, nf, nw, v
            else:
                step = max(step * 0.99, 1e-4)
        if cur > best[0]:
            best = (cur, atoms.copy(), w.copy())
    if best[1] is None:
        return None
    atoms, w = best[1], best[2]
    try:
        return certify([Fraction(float(a)).limit_denominator(10**9) for a in atoms], list(w), inst)
    except Rejected:
        return None


def search_best(inst: Instance, ks=(2, 3, 4), **kw) -> tuple[Certificate | None, int | None]:
    best, best_k = None, None
    for k in ks:
        c = local_search(inst, k, **kw)
        if c is not None and (best is None or c.value > best.value):
            best, best_k = c, k
    return best, best_k
