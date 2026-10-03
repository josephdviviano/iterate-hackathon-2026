"""Structure: 3 atoms {0, a, 1} (often 2: {a, 1} or {0, a}), a = (t - l) / k, integers k >= 1, l >= 0.
The weight on a is the one free parameter, maximized by grid plus golden-section search.
A 4-atom pass {0, a, b, 1}, b from sums j a + k b + l = t, never beat 3 atoms on train or random tests.
Free random search over 2-5 atom positions never beat this law for n <= 8.
Not settled: optimality (no proof for n >= 3), whether k = n and l + k = n are needed, large-n behaviour.
"""
from fractions import Fraction
import math
import time
from functools import lru_cache

_G = 200


def _vectors(n, k):
    out = []

    def rec(prefix, rem, slots):
        if slots == 1:
            c = prefix + (rem,)
            lc = math.lgamma(n + 1) - sum(math.lgamma(x + 1) for x in c)
            out.append((c, lc))
            return
        for x in range(rem + 1):
            rec(prefix + (x,), rem - x, slots - 1)

    rec((), n, k)
    return out


def _feasible_sets(n, atoms, t):
    """Count vectors with sum c_i a_i <= t, decided exactly."""
    vs = []
    for c, lc in _vectors(n, len(atoms)):
        if sum(ci * a for ci, a in zip(c, atoms)) <= t:
            vs.append((c, lc))
    return vs


def _value(vs, w):
    tot = 0.0
    for c, lc in vs:
        s = lc
        ok = True
        for ci, wi in zip(c, w):
            if ci:
                if wi <= 0:
                    ok = False
                    break
                s += ci * math.log(wi)
        if ok:
            tot += math.exp(s)
    return tot


def _full(inner, ws, m):
    """Weights for atoms inner + [0, 1] given interior weights, or None."""
    w1 = m - sum(w * a for w, a in zip(ws, inner))
    w0 = 1 - sum(ws) - w1
    if w1 < -1e-15 or w0 < -1e-15 or any(w < 0 for w in ws):
        return None
    return list(ws) + [max(w0, 0.0), max(w1, 0.0)]


def _optimize(inner, n, m, t, deadline):
    atoms = list(inner) + [Fraction(0), Fraction(1)]
    vs = _feasible_sets(n, atoms, t)
    fi = [float(a) for a in inner]
    mf = float(m)
    k = len(inner)

    def f(ws):
        w = _full(fi, ws, mf)
        return -1.0 if w is None else _value(vs, w)

    def interval(ws, i):
        # w_i range keeping other weights fixed and w0, w1 >= 0
        rest = [w for j, w in enumerate(ws) if j != i]
        ra = [a for j, a in enumerate(fi) if j != i]
        hi1 = (mf - sum(w * a for w, a in zip(rest, ra))) / fi[i]
        hi0 = (1 - mf - sum(w * (1 - a) for w, a in zip(rest, ra))) / (1 - fi[i])
        return 0.0, max(0.0, min(hi1, hi0))

    best_ws, best = [0.0] * k, f([0.0] * k)
    starts = [[0.0] * k]
    for i in range(k):
        s = [0.0] * k
        lo, hi = interval(s, i)
        s[i] = hi
        starts.append(s)
    for s in starts:
        ws = list(s)
        cur = f(ws)
        for _ in range(6 if k > 1 else 1):
            if time.time() > deadline:
                break
            for i in range(k):
                lo, hi = interval(ws, i)
                if hi <= lo:
                    ws[i] = 0.0
                    continue
                grid = [lo + (hi - lo) * j / _G for j in range(_G + 1)]
                vals = []
                for g in grid:
                    ws[i] = g
                    vals.append(f(ws))
                j = max(range(_G + 1), key=lambda j: vals[j])
                a, b = grid[max(j - 1, 0)], grid[min(j + 1, _G)]
                for _ in range(60):
                    c1, c2 = a + (b - a) * 0.382, a + (b - a) * 0.618
                    ws[i] = c1
                    v1 = f(ws)
                    ws[i] = c2
                    v2 = f(ws)
                    if v1 >= v2:
                        b = c2
                    else:
                        a = c1
                ws[i] = (a + b) / 2
                if f(ws) < vals[j]:
                    ws[i] = grid[j]
            nv = f(ws)
            if nv <= cur + 1e-13:
                cur = max(cur, nv)
                break
            cur = nv
        if cur > best:
            best, best_ws = cur, list(ws)
    return best, best_ws


def _candidates(n, m, t):
    out = set()
    for k in range(1, n + 1):
        for l in range(0, n):
            a = (t - l) / k
            if 0 < a < 1 and a != m:
                out.add(a)
    return sorted(out)


@lru_cache(maxsize=64)
def _search(n, m, t):
    """Ranked (value, interior atoms, weights) and whether the search finished in time."""
    deadline = time.time() + 3.5
    cands = _candidates(n, m, t)
    results = []
    for a in cands:
        v, ws = _optimize([a], n, m, t, deadline)
        results.append((v, [a], ws))
    results.sort(key=lambda r: -r[0])
    pairs = set()
    for _, (a,), _ in results[:3]:
        for b in cands:
            if b != a:
                pairs.add(tuple(sorted((a, b))))
        for j in range(1, n):
            for l in range(n):
                for k in range(1, n - j - l + 1):
                    b = (t - l - j * a) / k
                    if 0 < b < 1 and b != a:
                        pairs.add(tuple(sorted((a, b))))
    done = True
    for p in sorted(pairs):
        if time.time() > deadline:
            done = False
            break
        v, ws = _optimize(list(p), n, m, t, deadline)
        results.append((v, list(p), ws))
    # Prefer fewer atoms on float ties.
    results.sort(key=lambda r: -(r[0] - 1e-10 * len(r[1])))
    return results, done


def _exact(inner, ws, m):
    ws = [Fraction(w).limit_denominator(10**9) for w in ws]
    w1 = m - sum(w * a for w, a in zip(ws, inner))
    w0 = 1 - sum(ws) - w1
    if w1 < 0 or w0 < 0:
        return None
    return ws + [w0, w1]


def strategy(n, m, t):
    for v, inner, ws in _search(n, m, t)[0]:
        w = _exact(inner, ws, m)
        if w is not None:
            return list(inner) + [Fraction(0), Fraction(1)], w
    return [0, 1], [1 - m, m]


def confidence(n, m, t):
    if n <= 2:
        return 0.9
    _, done = _search(n, m, t)
    c = 0.8 if n <= 6 else 0.7 if n <= 8 else 0.55 if n <= 12 else 0.4
    return c if done else c - 0.15
