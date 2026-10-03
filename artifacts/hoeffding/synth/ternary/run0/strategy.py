"""Structure: at most three atoms {0, a, 1}, with a = (t - l) / k for integers k >= 1, l >= 0, l + k <= n.
The free weight on a is set by grid search and local refinement; the weights on 0 and 1 follow from the mean.
The optimum often puts zero weight on 0 or on 1, which gives two atoms {a, 1} or {0, a}.
A search over four atoms {0, a, b, 1}, with b from k a + j b + l = t, gave no gain on the train set.
Not settled: whether laws with more atoms or non-critical positions can do better, and so whether this is the supremum.
"""
from fractions import Fraction
import math


def _vectors(n, k):
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
    return out


def _objective(n, t, atoms):
    """Float evaluator of P(S_n <= t); the event set is decided exactly."""
    good = [(c, coef) for c, coef in _vectors(n, len(atoms))
            if sum(ci * a for ci, a in zip(c, atoms)) <= t]

    def f(w):
        s = 0.0
        for c, coef in good:
            term = coef
            for ci, wi in zip(c, w):
                if ci:
                    term *= wi ** ci
            s += term
        return s
    return f


def _full(head, hw, m):
    w1 = float(m) - sum(w * float(a) for w, a in zip(hw, head))
    w0 = 1.0 - sum(hw) - w1
    return w0, w1


def _optimize(n, m, t, head):
    """Maximize over head weights; atoms are head + [0, 1]."""
    atoms = list(head) + [Fraction(0), Fraction(1)]
    f = _objective(n, t, atoms)
    fm = float(m)

    def val(hw):
        if any(w < 0 for w in hw):
            return -1.0
        w0, w1 = _full(head, hw, m)
        if w0 < -1e-15 or w1 < -1e-15:
            return -1.0
        return f(list(hw) + [max(w0, 0.0), max(w1, 0.0)])

    d = len(head)
    best, bx = -1.0, None
    grid = 400 if d == 1 else 12
    caps = [min(1.0, fm / float(a)) for a in head]

    def rec(prefix, i):
        nonlocal best, bx
        if i == d:
            v = val(prefix)
            if v > best:
                best, bx = v, list(prefix)
            return
        for g in range(grid + 1):
            rec(prefix + [caps[i] * g / grid], i + 1)

    rec([], 0)
    if d == 0:
        return best, atoms, []
    step = [c / grid for c in caps]
    while max(step) > 1e-12:
        moved = False
        for i in range(d):
            for s in (step[i], -step[i]):
                x = list(bx)
                x[i] += s
                v = val(x)
                if v > best + 1e-16:
                    best, bx, moved = v, x, True
        if not moved:
            step = [s / 2 for s in step]
    return best, atoms, bx


def _candidates(n, t):
    out = set()
    for k in range(1, n + 1):
        for l in range(0, n):
            a = (t - l) / k
            if 0 < a < 1 and l + k <= n:
                out.add(a)
    return sorted(out)


def strategy(n, m, t):
    best = (-1.0, [Fraction(0), Fraction(1)], [])
    for head in [[]] + [[a] for a in _candidates(n, t)]:
        r = _optimize(n, m, t, head)
        if r[0] > best[0]:
            best = r
    _, atoms, hw = best
    head = atoms[:-2]
    hw = [Fraction(w).limit_denominator(10**12) for w in hw]
    while True:
        w1 = m - sum((w * a for w, a in zip(hw, head)), Fraction(0))
        w0 = 1 - sum(hw, Fraction(0)) - w1
        if w0 >= 0 and w1 >= 0:
            return atoms, hw + [w0, w1]
        hw = [w * (1 - Fraction(1, 10**11)) for w in hw]
