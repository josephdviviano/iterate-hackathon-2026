"""Meester's (2008) conjectured extremal families, as certified candidates.

Binary: support {a_j, b_j}, j = 0 .. ceil(t/m) - 1, with the mass on b_j
fixed by the mean. Ternary: support {0, a, 1} with a = (t - l) / k for
integers k >= 1, l >= 0, k + l <= n - 1, l < t < l + k; one free weight,
optimized numerically and then certified. For n = 2 the supports are proven
extremal, so the best of these is the exact supremum.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .problem import Instance
from .verify import Certificate, Rejected, certify


def binary_candidates(inst: Instance) -> list[Certificate]:
    n, m, t = inst.n, inst.m, inst.t
    out = []
    for j in range(0, math.ceil(t / m)):
        if j <= t:
            a, b = (t - j) / (n - j), Fraction(1)
            pi = 1 - (1 - m) * (n - j) / (n - t)
        else:
            a, b = Fraction(0), t / j
            pi = j * m / t
        if not (0 <= a < m < b <= 1 and 0 <= pi <= 1):
            continue
        try:
            out.append(certify([a, b], [1 - pi, pi], inst))
        except Rejected:
            continue
    return out


def _ternary_value(a: Fraction, p: Fraction, inst: Instance) -> Certificate | None:
    m = inst.m
    q = (m - p) / a
    r = 1 - p - q
    if p < 0 or q < 0 or r < 0:
        return None
    return certify([Fraction(0), a, Fraction(1)], [r, q, p], inst)


def ternary_candidates(inst: Instance, iters: int = 60) -> list[Certificate]:
    n, m, t = inst.n, inst.m, inst.t
    out = []
    for k in range(1, n):
        for l in range(0, n - k):
            if not (l < t < l + k):
                continue
            a = (t - l) / k
            if not (0 < a < 1):
                continue
            # p = P(X = 1) in [0, p_max], golden section on the exact value, then certify at a rational p.
            p_max = min(m, (m - a) / (1 - a)) if a < m else m  # keeps q, r >= 0
            p_max = max(Fraction(0), p_max)
            lo, hi = 0.0, float(p_max)
            f = lambda p: float(c.value) if (c := _ternary_value(a, Fraction(p).limit_denominator(10**9), inst)) else -1.0
            g = (math.sqrt(5) - 1) / 2
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            f1, f2 = f(x1), f(x2)
            for _ in range(iters):
                if f1 < f2:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + g * (hi - lo)
                    f2 = f(x2)
                else:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - g * (hi - lo)
                    f1 = f(x1)
            best = None
            for p in (lo, hi, (lo + hi) / 2, 0.0, float(p_max)):
                c = _ternary_value(a, Fraction(p).limit_denominator(10**9), inst)
                if c is not None and (best is None or c.value > best.value):
                    best = c
            if best is not None:
                out.append(best)
    return out


def family_best(inst: Instance) -> tuple[Certificate, str]:
    cands = [(c, "binary") for c in binary_candidates(inst)] + [(c, "ternary") for c in ternary_candidates(inst)]
    if not cands:
        raise Rejected("no family candidate")
    return max(cands, key=lambda cf: cf[0].value)


def exact_n2(inst: Instance) -> Certificate:
    """Meester (2008): for n = 2 the extremal support is one of the family supports."""
    assert inst.n == 2
    return family_best(inst)[0]


def exact_n1_cert(inst: Instance) -> Certificate:
    from .problem import exact_n1
    _, atoms, weights = exact_n1(inst)
    return certify(atoms, weights, inst)
