"""Two-tier verification: a float estimate decides whether the exact certificate is needed.

The decision that matters for a candidate is whether it can change the
certified best of its instance. The float tier returns the mass strictly inside
the event, the mass within TIE of the boundary including exactly on it (where
only exact arithmetic decides the side), and a conservative rounding margin.
A float sum of exactly 0 is NOT an exact tie: the laws agents write break
ties with perturbations below float resolution, so a float 0 can be a true
negative. The exact value lies in [value, value + tie_mass + margin]. The policy only
spends or saves certification compute: a wrong skip can lose a candidate,
never corrupt a bound, since every reported bound is certified. The policy certifies a candidate
when that upper end reaches the incumbent, skips it otherwise, and audits a
random fraction of skips. The reward and
every reported bound come from the exact tier only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction

from .verify import _rational, count_vectors

EPS = 2.0**-52
TIE = 1e-9
WIN_TOL = 1e-9  # a candidate changes the best only if it exceeds the incumbent by more than this


@dataclass
class Estimate:
    value: float      # mass strictly inside the event, away from the boundary
    margin: float     # rounding allowance
    tie_mass: float   # mass within TIE of the boundary; the exact tier decides its side
    n_ops: int

    @property
    def tie(self) -> bool:
        return self.tie_mass > 0

    @property
    def upper(self) -> float:
        return self.value + self.tie_mass + self.margin


def _floats(xs) -> list[float]:
    return [float(_rational(x)) for x in xs]


def estimate_linear(atoms, weights, inst) -> Estimate:
    a, w = _floats(atoms), _floats(weights)
    tot = sum(w)
    w = [x / tot for x in w]
    d = {0.0: 1.0}
    ops = 0
    for ci in inst.c:
        nd: dict = {}
        for s, p in d.items():
            for ai, wi in zip(a, w):
                k = s + ci * ai
                nd[k] = nd.get(k, 0.0) + p * wi
                ops += 1
        d = nd
    lt = sum(p for s, p in d.items() if s < -TIE)
    tie_mass = sum(p for s, p in d.items() if abs(s) <= TIE)
    return Estimate(lt, 8 * ops * EPS * max(1.0, lt), tie_mass, ops)


def estimate_hoeffding(atoms, weights, inst) -> Estimate:
    from .verify import repair
    ra = [_rational(x) for x in atoms]
    rw = [_rational(x) for x in weights]
    rw, _ = repair(ra, rw, inst.m)  # exact and cheap: two linear equations
    a, w = [float(x) for x in ra], [float(x) for x in rw]
    t = float(inst.t)
    tot, tie_mass, ops = 0.0, 0.0, 0
    logw = [math.log(x) if x > 0 else -math.inf for x in w]
    for c, coef in count_vectors(inst.n, len(a)):
        s = sum(ci * ai for ci, ai in zip(c, a))
        ops += len(a)
        if s <= t + TIE:
            e = sum(ci * lw for ci, lw in zip(c, logw) if ci)
            term = coef * math.exp(e) if e > -math.inf else 0.0
            if abs(s - t) <= TIE:
                tie_mass += term
            else:
                tot += term
    return Estimate(tot, 8 * ops * EPS * max(1.0, tot), tie_mass, ops)


@dataclass
class Decision:
    action: str  # "certify", "skip", "audit"
    estimate: Estimate
    incumbent: float | None


def decide(est: Estimate, incumbent: float | None, audit_rate: float, rng: random.Random) -> Decision:
    if incumbent is None or est.upper > incumbent + WIN_TOL:
        return Decision("certify", est, incumbent)
    if rng.random() < audit_rate:
        return Decision("audit", est, incumbent)
    return Decision("skip", est, incumbent)


@dataclass
class Counters:
    requested: int = 0
    certified: int = 0
    skipped: int = 0
    audited: int = 0
    audit_disagreements: int = 0
    exact_seconds: float = 0.0
    estimate_seconds: float = 0.0
    skipped_exact_seconds_est: float = 0.0  # what the skipped certificates would have cost, from timed audits

    def to_json(self) -> dict:
        return self.__dict__ | {"fraction_certified": (self.certified + self.audited) / max(1, self.requested)}
