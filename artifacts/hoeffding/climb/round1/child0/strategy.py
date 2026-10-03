"""Structure: 2 atoms {x, y}, x < m < y, both from T = {0, 1} U {(t - k)/c : k, c >= 0 integers, k + c <= n}.
Sometimes 3 atoms {0, y, 1}, y in T, win (n=6, m=3/5, t=9/10): the atom at 1 only carries mean.
Search: all pairs of T, and all {0, y, 1}; y deduplicated by its inclusion profile, weights in floats.
Breakpoint local search with 4 atoms (n <= 6) and 3 free atoms (n <= 20) found nothing better.
Not settled: optimality; 5+ atom laws for n >= 4 were not searched.
"""
import math
from fractions import Fraction


def _tight_set(n, t):
    T = {Fraction(0), Fraction(1)}
    for k in range(n):
        for c in range(1, n - k + 1):
            x = (t - k) / c
            if 0 <= x <= 1:
                T.add(x)
    return sorted(T)


def _two_atom(n, m, t, T, C):
    best = (-1.0, None, None)
    for x in T:
        if x >= m:
            break
        for y in T:
            if y <= m:
                continue
            if n * x > t:
                break
            j = min(n, math.floor((t - n * x) / (y - x)))
            q = float((m - x) / (y - x))
            v = sum(C[n][i] * q ** i * (1 - q) ** (n - i) for i in range(j + 1))
            if v > best[0]:
                best = (v, [x, y], [1 - q, q])
    return best


def _three_atom(n, m, t, T, C):
    """Laws on {0, y, 1}; for each count c1 of ones at most L[c1] atoms y fit under t."""
    profiles = {}
    for y in T:
        if y in (0, 1):
            continue
        L = tuple(min(n - c1, math.floor((t - c1) / y)) for c1 in range(min(n, math.floor(t)) + 1))
        if L not in profiles or y > profiles[L]:
            profiles[L] = y
    fm = float(m)
    best = (-1.0, None, None)
    for L, y in profiles.items():
        fy = float(y)

        def f(u):
            w1 = fm - u * fy
            w0 = 1 - u - w1
            s = 0.0
            for c1, lim in enumerate(L):
                r = n - c1
                s += C[n][c1] * w1 ** c1 * sum(C[r][cy] * u ** cy * w0 ** (r - cy) for cy in range(lim + 1))
            return s
        hi = min(fm / fy, (1 - fm) / (1 - fy))
        G = 24
        us = [hi * g / G for g in range(G + 1)]
        vs = [f(u) for u in us]
        g = max(range(G + 1), key=vs.__getitem__)
        a, b = us[max(g - 1, 0)], us[min(g + 1, G)]
        for _ in range(40):
            c1, c2 = a + (b - a) * 0.382, a + (b - a) * 0.618
            if f(c1) < f(c2):
                a = c1
            else:
                b = c2
        u = (a + b) / 2
        v = f(u)
        if vs[g] >= v:
            u, v = us[g], vs[g]
        if v > best[0]:
            w1 = fm - u * fy
            best = (v, [y, Fraction(0), Fraction(1)], [u, max(1 - u - w1, 0.0), max(w1, 0.0)])
    return best


def _search(n, m, t):
    m, t = Fraction(m), Fraction(t)
    C = [[math.comb(r, i) for i in range(r + 1)] for r in range(n + 1)]
    T = _tight_set(n, t)
    return max(_two_atom(n, m, t, T, C), _three_atom(n, m, t, T, C), key=lambda b: b[0])


def strategy(n, m, t):
    v, atoms, w = _search(n, m, t)
    if atoms is None:
        return [0, 1], [1 - m, m]
    w = [Fraction(x).limit_denominator(10 ** 12) for x in w]
    keep = [i for i in range(len(atoms)) if w[i] > 0 or i >= len(atoms) - 2]
    return [atoms[i] for i in keep], [w[i] for i in keep]


def confidence(n, m, t):
    v, atoms, w = _search(n, m, t)
    base = {3: 0.7, 4: 0.6, 5: 0.55, 6: 0.5}.get(n, 0.4 if n > 6 else 0.9)
    if atoms is not None and len(atoms) == 3 and min(w) > 1e-9:
        base -= 0.1
    return base
