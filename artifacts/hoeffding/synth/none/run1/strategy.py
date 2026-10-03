"""Structure: the best laws found have 2 or 3 atoms, all on the grid {0, 1} U {(t - i)/j}.
Usually {(t - i)/j, 1} (an atom exactly at a threshold, the remaining mass at 1); for large m and t
the law is {0, x} with x = t/j or (t - i)/j, sometimes with a small atom at 1 added (3 atoms).
Exhaustive 4-atom grid search, a second-level grid (thresholds that include interior atoms) and
random continuous search found no better law. Not settled: optimality for n >= 5 and any 4+ atom law.
"""
import itertools
import time
from fractions import Fraction

from verify import count_vectors

_BUDGET = 3.5
_last = {}


def _grid(n, t):
    g = {Fraction(0), Fraction(1)}
    for i in range(n + 1):
        for j in range(1, n + 1):
            a = (t - i) / j
            if 0 <= a <= 1:
                g.add(a)
    return sorted(g)


def _terms(n, atoms, t):
    return [(coef, c) for c, coef in count_vectors(n, len(atoms))
            if sum(ci * a for ci, a in zip(c, atoms)) <= t]


def _value(terms, w):
    s = 0.0
    for coef, c in terms:
        p = coef
        for ci, wi in zip(c, w):
            if ci:
                p *= wi ** ci
        s += p
    return s


def _two(atoms, m):
    a, b = atoms
    wb = (m - a) / (b - a)
    return [1 - wb, wb]


def _three(atoms, m, u):
    """Weights with u on the middle atom; the outer two absorb the mean constraint."""
    a, b, c = atoms
    wc = (m - a - u * (b - a)) / (c - a)
    return [1 - u - wc, u, wc]


def _opt3(terms, atoms, m):
    a, b, c = atoms
    hi = min((c - m) / (c - b), (m - a) / (b - a), 1.0)
    if hi <= 0:
        return -1.0, None
    f = lambda u: _value(terms, _three(atoms, m, u))
    N = 16
    us = [hi * k / N for k in range(N + 1)]
    vs = [f(u) for u in us]
    k = max(range(N + 1), key=vs.__getitem__)
    lo, up = us[max(k - 1, 0)], us[min(k + 1, N)]
    for _ in range(60):
        x1, x2 = lo + (up - lo) * 0.382, lo + (up - lo) * 0.618
        if f(x1) < f(x2):
            lo = x1
        else:
            up = x2
    u = (lo + up) / 2
    v = f(u)
    if v < vs[k]:
        u, v = us[k], vs[k]
    return v, _three(atoms, m, u)


def _search(n, m, t):
    start = time.time()
    mf = float(m)
    G = _grid(n, t)
    best = (-1.0, [Fraction(0), Fraction(1)], [1 - mf, mf])
    complete = True

    def consider(S):
        nonlocal best
        if not (S[0] <= m <= S[-1]) or S[0] == m or S[-1] == m:
            return
        fa = [float(a) for a in S]
        terms = _terms(n, S, t)
        if len(S) == 2:
            v, w = _value(terms, _two(fa, mf)), _two(fa, mf)
        else:
            v, w = _opt3(terms, fa, mf)
        if w is not None and v > best[0] + 1e-13:
            best = (v, list(S), w)

    for S in itertools.combinations(G, 2):
        consider(S)
    zero, one = G[0], G[-1]
    inner = G[1:-1]
    for x in inner:
        consider((zero, x, one))
    for x, y in itertools.combinations(inner, 2):
        if time.time() - start > _BUDGET:
            complete = False
            break
        consider((zero, x, y))
        consider((x, y, one))
    else:
        for S in itertools.combinations(inner, 3):
            if time.time() - start > _BUDGET:
                complete = False
                break
            consider(S)
    return best, complete


def strategy(n, m, t):
    (v, atoms, w), complete = _search(n, Fraction(m), Fraction(t))
    _last[(n, m, t)] = (v, complete, len(atoms))
    return atoms, w


def confidence(n, m, t):
    key = (n, Fraction(m), Fraction(t))
    if key not in _last:
        strategy(*key)
    v, complete, k = _last[key]
    p = 0.8 if n <= 4 else 0.7 if n <= 6 else 0.55
    if not complete:
        p -= 0.2
    if k == 3:
        p -= 0.05
    return max(0.05, p)
