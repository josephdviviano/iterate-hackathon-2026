"""Exact certificate of a candidate measure.

value = sum over count vectors c (c_1 + ... + c_k = n, sum c_i a_i <= t) of
multinomial(n; c) * prod w_i^c_i, in Fractions. A candidate with atoms outside
[0, 1], a negative weight, or a mean that cannot be repaired is rejected.
Floats are rounded to rationals with denominator <= 10^12 first. The last two
weights are then re-solved so that sum w = 1 and sum w a = m hold exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from .problem import Instance

MAX_DENOMINATOR = 10**12


@lru_cache(maxsize=None)
def count_vectors(n: int, k: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    """All (c, multinomial(n; c)) with c of length k summing to n."""
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            c = prefix + (remaining,)
            coef = math.factorial(n)
            for ci in c:
                coef //= math.factorial(ci)
            out.append((c, coef))
            return
        for x in range(remaining + 1):
            rec(prefix + (x,), remaining - x, slots - 1)

    rec((), n, k)
    return tuple(out)


@dataclass
class Certificate:
    atoms: list[Fraction]
    weights: list[Fraction]
    value: Fraction
    repaired: bool

    def to_json(self) -> dict:
        return {"atoms": [str(a) for a in self.atoms], "weights": [str(w) for w in self.weights],
                "value": str(self.value), "value_float": float(self.value), "repaired": self.repaired}


class Rejected(ValueError):
    pass


def _rational(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise Rejected(f"non-finite number {x}")
        return Fraction(x).limit_denominator(MAX_DENOMINATOR)
    raise Rejected(f"unsupported number type {type(x).__name__}")


def repair(atoms: list[Fraction], weights: list[Fraction], m: Fraction) -> tuple[list[Fraction], bool]:
    """Re-solve the last two weights so that sum w = 1 and sum w a = m exactly."""
    k = len(atoms)
    if k == 1:
        if atoms[0] != m:
            raise Rejected("a single atom must sit at m")
        return [Fraction(1)], weights != [Fraction(1)]
    a1, a2 = atoms[-2], atoms[-1]
    if a1 == a2:
        raise Rejected("the last two atoms coincide; cannot repair")
    head = weights[:-2]
    W = sum(head, Fraction(0))
    M = sum((w * a for w, a in zip(head, atoms[:-2])), Fraction(0))
    # w1 + w2 = 1 - W ; w1 a1 + w2 a2 = m - M
    w2 = ((m - M) - a1 * (1 - W)) / (a2 - a1)
    w1 = (1 - W) - w2
    new = head + [w1, w2]
    if any(w < 0 for w in new):
        raise Rejected("repair gives a negative weight: the candidate cannot have mean m")
    return new, new != weights


def certify(atoms, weights, inst: Instance) -> Certificate:
    atoms = [_rational(a) for a in atoms]
    weights = [_rational(w) for w in weights]
    if not atoms or len(atoms) != len(weights):
        raise Rejected("atoms and weights must be non-empty and of equal length")
    if len(set(atoms)) != len(atoms):
        raise Rejected("atoms must be distinct")
    if any(a < 0 or a > 1 for a in atoms):
        raise Rejected("every atom must lie in [0, 1]")
    if any(w < 0 for w in weights):
        raise Rejected("every weight must be non-negative")
    weights, repaired = repair(atoms, weights, inst.m)
    keep = [i for i, w in enumerate(weights) if w > 0]
    atoms = [atoms[i] for i in keep]
    weights = [weights[i] for i in keep]
    value = tail_probability(atoms, weights, inst.n, inst.t)
    return Certificate(atoms, weights, value, repaired)


def tail_probability(atoms: list[Fraction], weights: list[Fraction], n: int, t: Fraction) -> Fraction:
    """P(S_n <= t) for the discrete measure, exactly."""
    total = Fraction(0)
    for c, coef in count_vectors(n, len(atoms)):
        if sum((ci * a for ci, a in zip(c, atoms)), Fraction(0)) <= t:
            term = Fraction(coef)
            for ci, w in zip(c, weights):
                if ci:
                    term *= w**ci
            total += term
    return total
