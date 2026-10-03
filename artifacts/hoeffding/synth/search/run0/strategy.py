"""Structure: atoms sit on the grid {0, 1} U {(t - r)/j : 1 <= j <= n, 0 <= r < n}.
Best law is mostly two atoms: {(t - r)/(n - r), 1} (r ones allowed), or {0, t/j} when j copies fit.
Sometimes three atoms {0, t/j, 1} win by a small margin (e.g. n=6, m=3/5, t=9/10).
On the train set no 4-atom law on this grid beat the best 3-atom law.
Not settled: whether off-grid atoms or k >= 4 laws can beat this, i.e. if it is the supremum.
"""
import itertools
import math
import time
from fractions import Fraction
from functools import lru_cache

EPS = 1e-9
BUDGET = 4.0
GOLDEN = (math.sqrt(5) - 1) / 2


@lru_cache(maxsize=None)
def _vectors(n, k):
    out = []

    def rec(prefix, rem, slots):
        if slots == 1:
            c = prefix + (rem,)
            coef = math.factorial(n)
            for ci in c:
                coef //= math.factorial(ci)
            out.append((c, coef))
            return
        for x in range(rem + 1):
            rec(prefix + (x,), rem - x, slots - 1)

    rec((), n, k)
    return tuple(out)


def _grid(n, t):
    g = {Fraction(0), Fraction(1)}
    for j in range(1, n + 1):
        for r in range(n):
            a = (t - r) / j
            if 0 < a < 1:
                g.add(a)
    return sorted(g)


def _feasible(atoms, n, t):
    return [(c, coef) for c, coef in _vectors(n, len(atoms))
            if sum(ci * a for ci, a in zip(c, atoms)) <= t + EPS]


def _value(feas, w):
    total = 0.0
    for c, coef in feas:
        p = coef
        for ci, wi in zip(c, w):
            if ci:
                p *= wi ** ci
        total += p
    return total


def _three_point(lo, mid, hi, m, h):
    """Weights on (lo, mid, hi) with weight h on mid and mean m."""
    w_hi = (m - lo - h * (mid - lo)) / (hi - lo)
    return [1 - h - w_hi, h, w_hi]


def _best_mid_weight(atoms, feas, m):
    lo, mid, hi = atoms
    # admissible h keeps both outer weights non-negative
    h_max = min((m - lo) / (mid - lo), (hi - m) / (hi - mid), 1.0)
    if h_max <= 0:
        return 0.0, -1.0
    f = lambda h: _value(feas, _three_point(lo, mid, hi, m, h))
    steps = 24
    vals = [f(h_max * i / steps) for i in range(steps + 1)]
    i = max(range(steps + 1), key=vals.__getitem__)
    a, b = h_max * max(i - 1, 0) / steps, h_max * min(i + 1, steps) / steps
    x1, x2 = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(40):
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + GOLDEN * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - GOLDEN * (b - a)
            f1 = f(x1)
    cands = [(vals[i], h_max * i / steps), (f1, x1), (f2, x2)]
    v, h = max(cands)
    return h, v


def _complete(atoms, head, m):
    """Weights with `head` on all but the last two atoms, which are solved for mean m."""
    if min(head, default=0) < 0:
        return None
    rest = 1 - sum(head)
    mean = sum(h * a for h, a in zip(head, atoms))
    a1, a2 = atoms[-2], atoms[-1]
    w2 = ((m - mean) - a1 * rest) / (a2 - a1)
    w1 = rest - w2
    if w1 < 0 or w2 < 0:
        return None
    return list(head) + [w1, w2]


def _nelder_mead(f, x0, step, iters):
    d = len(x0)
    pts = [list(x0)] + [[x0[j] + (step if j == i else 0) for j in range(d)] for i in range(d)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(d + 1), key=lambda i: -vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        c = [sum(p[j] for p in pts[:-1]) / d for j in range(d)]
        r = [2 * c[j] - pts[-1][j] for j in range(d)]
        vr = f(r)
        if vr > vals[0]:
            e = [3 * c[j] - 2 * pts[-1][j] for j in range(d)]
            ve = f(e)
            pts[-1], vals[-1] = (e, ve) if ve > vr else (r, vr)
        elif vr > vals[-2]:
            pts[-1], vals[-1] = r, vr
        else:
            cc = [(c[j] + pts[-1][j]) / 2 for j in range(d)]
            vc = f(cc)
            if vc > vals[-1]:
                pts[-1], vals[-1] = cc, vc
            else:
                for i in range(1, d + 1):
                    pts[i] = [(pts[0][j] + pts[i][j]) / 2 for j in range(d)]
                    vals[i] = f(pts[i])
    i = max(range(d + 1), key=vals.__getitem__)
    return vals[i], pts[i]


def _add_atom(atoms, weights, x, n, m, t):
    """Optimise weights after adding atom x; the outer two atoms stay last."""
    order = sorted(atoms + [x])
    new = order[1:-1] + [order[0], order[-1]]
    fa = [float(a) for a in new]
    feas = _feasible(fa, n, float(t))
    old = {a: float(w) for a, w in zip(atoms, weights)}
    x0 = [old.get(a, 0.0) for a in new[:-2]]

    def f(h):
        w = _complete(fa, h, float(m))
        return _value(feas, w) if w else -1.0

    v, h = _nelder_mead(f, x0, 0.02, 60 * len(x0))
    return v, new, h


def _exact(atoms, weights, n, t):
    total = Fraction(0)
    for c, coef in _vectors(n, len(atoms)):
        if sum((ci * a for ci, a in zip(c, atoms)), Fraction(0)) <= t:
            term = Fraction(coef)
            for ci, wi in zip(c, weights):
                if ci:
                    term *= wi ** ci
            total += term
    return total


def strategy(n, m, t):
    start = time.time()
    m, t = Fraction(m), Fraction(t)
    grid = _grid(n, t)
    mf, tf = float(m), float(t)

    best_val, best = Fraction(-1), None
    for lo, hi in itertools.combinations(grid, 2):
        if lo <= m <= hi and lo < hi:
            w_hi = (m - lo) / (hi - lo)
            law = ([lo, hi], [1 - w_hi, w_hi])
            v = _exact(*law, n, t)
            if v > best_val:
                best_val, best = v, law

    scored = []
    for atoms in itertools.combinations(grid, 3):
        if not atoms[0] < m < atoms[2]:
            continue
        if time.time() - start > BUDGET / 2:
            break
        fa = [float(a) for a in atoms]
        h, v = _best_mid_weight(fa, _feasible(fa, n, tf), mf)
        if v > float(best_val) + 1e-12:
            scored.append((v, atoms, h))
    scored.sort(key=lambda s: -s[0])
    for _, (lo, mid, hi), h in scored[:5]:
        wm = Fraction(h).limit_denominator(10**12)
        w_hi = (m - lo - wm * (mid - lo)) / (hi - lo)
        ws = [1 - wm - w_hi, wm, w_hi]
        if min(ws) < 0:
            continue
        v = _exact([lo, mid, hi], ws, n, t)
        if v > best_val:
            best_val, best = v, ([mid, lo, hi], [ws[1], ws[0], ws[2]])

    while len(best[0]) < 4 and time.time() - start < BUDGET:
        found = None
        for x in grid:
            if x in best[0] or time.time() - start > BUDGET:
                continue
            v, atoms, h = _add_atom(best[0], best[1], x, n, m, t)
            if v > float(best_val) + 1e-12 and (found is None or v > found[0]):
                found = (v, atoms, h)
        if found is None:
            break
        _, atoms, h = found
        head = [Fraction(x).limit_denominator(10**12) for x in h]
        ws = _complete(atoms, head, m)
        if ws is None:
            break
        v = _exact(atoms, ws, n, t)
        if v <= best_val:
            break
        best_val, best = v, (atoms, ws)
    return best
