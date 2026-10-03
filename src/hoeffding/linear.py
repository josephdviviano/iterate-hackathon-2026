"""Linear inequalities between iid variables (Bellec and Fritz 2024; Problem 6.39 of Tao et al.).

C(c) = sup over iid laws on [0, infinity) of P(sum_i c_i X_i < 0). The event is
scale-invariant, so atoms are taken in [0, 1]. A candidate is a discrete law;
its value is certified exactly by convolving the law of each c_i X_i.
Known: c = (2, -1, -1), i.e. P(2 X1 < X2 + X3), has C = 2/3 (proven, deck 1, 2, 4, ...);
c = (1, 1, 1, -2), i.e. P(X1 + X2 + X3 < 2 X4), has 0.400695 <= C <= 0.417 (lower
bound conjectured exact). Other c: no published value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .verify import Certificate, Rejected, _rational

KNOWN = {
    (2, -1, -1): (Fraction(2, 3), Fraction(2, 3), "proven (Bellec-Fritz)"),
    (1, 1, 1, -2): (Fraction(400695, 10**6), Fraction(417, 1000), "Bellec-Fritz 2024, lower conjectured exact"),
}


@dataclass(frozen=True)
class LinearInstance:
    c: tuple[int, ...]

    @property
    def key(self) -> str:
        return "c" + "_".join(str(x) for x in self.c)

    def to_json(self) -> dict:
        return {"c": list(self.c), "args": [list(self.c)]}

    @staticmethod
    def from_json(d: dict) -> "LinearInstance":
        return LinearInstance(tuple(int(x) for x in d["c"]))


def value_of(atoms: list[Fraction], weights: list[Fraction], c: tuple[int, ...]) -> tuple[Fraction, Fraction]:
    """(P(sum c_i X_i < 0), P(sum c_i X_i = 0)) exactly.

    Atoms and weights are put on common denominators and the convolution runs over
    Python integers; the result is divided back once. Same value as the Fraction
    convolution, about 14x faster, since no gcd is taken per operation."""
    da = 1
    for a in atoms:
        da = da * a.denominator // math.gcd(da, a.denominator)
    dw = 1
    for w in weights:
        dw = dw * w.denominator // math.gcd(dw, w.denominator)
    A = [int(a * da) for a in atoms]
    W = [int(w * dw) for w in weights]
    d = {0: 1}
    for ci in c:
        nd: dict = {}
        for s, p in d.items():
            for a, w in zip(A, W):
                key = s + ci * a
                nd[key] = nd.get(key, 0) + p * w
        d = nd
    scale = Fraction(1, dw ** len(c))
    lt = sum((p for s, p in d.items() if s < 0), 0) * scale
    eq = d.get(0, 0) * scale
    return lt, eq


MAX_ATOMS = 64


def certify_linear(atoms, weights, inst: LinearInstance, max_atoms: int | None = None) -> Certificate:
    atoms = [_rational(a) for a in atoms]
    weights = [_rational(w) for w in weights]
    if not atoms or len(atoms) != len(weights):
        raise Rejected("atoms and weights must be non-empty and of equal length")
    if len(set(atoms)) != len(atoms):
        raise Rejected("atoms must be distinct")
    if any(a < 0 or a > 1 for a in atoms):
        raise Rejected("every atom must lie in [0, 1]; the event is scale-invariant")
    if any(w < 0 for w in weights):
        raise Rejected("every weight must be non-negative")
    total = sum(weights, Fraction(0))
    if total == 0:
        raise Rejected("weights sum to zero")
    repaired = total != 1
    weights = [w / total for w in weights]  # repair: normalise
    keep = [i for i, w in enumerate(weights) if w > 0]
    atoms = [atoms[i] for i in keep]
    weights = [weights[i] for i in keep]
    cap = max_atoms if max_atoms is not None else MAX_ATOMS
    if len(atoms) > cap:
        raise Rejected(f"at most {cap} atoms")
    lt, _ = value_of(atoms, weights, inst.c)
    return Certificate(atoms, weights, lt, repaired)


def naive_value(inst: LinearInstance) -> float:
    """Best two-atom law {0, 1}, by a grid over the weight. The naive start."""
    best = 0.0
    for k in range(1, 100):
        p = Fraction(k, 100)
        lt, _ = value_of([Fraction(0), Fraction(1)], [1 - p, p], inst.c)
        best = max(best, float(lt))
    return best


def upper_wall(inst: LinearInstance) -> float | None:
    k = KNOWN.get(inst.c)
    return float(k[1]) if k else None


def bellec_fritz_law(p: Fraction, N: int) -> tuple[list[Fraction], list[Fraction]]:
    """nu = p d_0 + (1 - p)/N sum_{i=1..N} d_{1 - 2^-i}; the seed construction for (1, 1, 1, -2)."""
    return [Fraction(0)] + [1 - Fraction(1, 2**i) for i in range(1, N + 1)], [p] + [(1 - p) / N] * N


TRAIN = [LinearInstance(c) for c in [(1, 1, 1, -2), (1, 1, 1, 1, -3), (1, 2, -3)]]
FOCUS = [LinearInstance((1, 1, 1, -2))]
TEST = [LinearInstance(c) for c in [(2, -1, -1), (1, 1, -2), (1, 1, 1, 1, 1, -4), (2, 1, 1, -3)]]
