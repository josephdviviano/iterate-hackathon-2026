"""Instances, the two known walls, and the exact n = 1 answer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Instance:
    n: int
    m: Fraction
    t: Fraction

    def __post_init__(self):
        if not (0 < self.m < 1):
            raise ValueError("need 0 < m < 1")
        if not (0 <= self.t < self.n * self.m):
            raise ValueError("need 0 <= t < n m, otherwise the supremum is 1")

    @property
    def key(self) -> str:
        return f"n{self.n}_m{self.m.numerator}-{self.m.denominator}_t{self.t.numerator}-{self.t.denominator}"

    def to_json(self) -> dict:
        return {"n": self.n, "m": str(self.m), "t": str(self.t)}

    @staticmethod
    def from_json(d: dict) -> "Instance":
        return Instance(int(d["n"]), Fraction(d["m"]), Fraction(d["t"]))


def bernoulli_value(inst: Instance) -> Fraction:
    """P(Bin(n, m) <= t). Hoeffding's own two-point measure; the baseline every method must beat."""
    n, m = inst.n, inst.m
    k = math.floor(inst.t)
    return sum(Fraction(math.comb(n, j)) * m**j * (1 - m) ** (n - j) for j in range(k + 1))


def hoeffding_bound(inst: Instance) -> float:
    """Hoeffding (1963) Theorem 1 for the lower tail: exp(-n kl(t/n || m)). The rigorous upper wall."""
    n, m, t = inst.n, float(inst.m), float(inst.t)
    a = t / n
    if a == 0:
        return (1 - m) ** n
    kl = a * math.log(a / m) + (1 - a) * math.log((1 - a) / (1 - m))
    return math.exp(-n * kl)


def exact_n1(inst: Instance) -> tuple[Fraction, list[Fraction], list[Fraction]]:
    """n = 1: Markov on 1 - X gives (1 - m) / (1 - t), attained by atoms {t, 1}."""
    assert inst.n == 1
    m, t = inst.m, inst.t
    return (1 - m) / (1 - t), [t, Fraction(1)], [(1 - m) / (1 - t), (m - t) / (1 - t)]


def grid(ns: list[int], ms: list[Fraction], fracs: list[Fraction]) -> list[Instance]:
    """t = frac * n m, so every instance has 0 <= t < n m."""
    return [Instance(n, m, f * n * m) for n in ns for m in ms for f in fracs]


MS = [Fraction(1, 5), Fraction(2, 5), Fraction(3, 5)]
FRACS = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]
TRAIN = grid([3, 4, 6], MS, FRACS)
TEST = grid([1, 2, 5, 8, 10], MS, FRACS)
