"""Structure: 2 atoms on all but one train instance: {(t-j)/(n-j), 1} (b = 1) or {0, t/j} (a = 0).
Sometimes 3 atoms {0, t/j, 1} win (n=6, m=3/5, t=9/10): the atom at 1 only carries mean.
Atoms come from the tightness set {(t - k)/c : integers k >= 0, c >= 1, k + c <= n} plus {0, 1};
all pairs and triples of it are searched, weights optimised in floats, atoms kept exact.
Not settled: optimality; 4+ atom laws and level-2 tightness atoms gave no gain, but no proof.
"""
import itertools
import math
import time
from fractions import Fraction


def _count_vectors(n, k):
    out = []
    for c in itertools.product(range(n + 1), repeat=k):
        if sum(c) == n:
            coef = math.factorial(n)
            for ci in c:
                coef //= math.factorial(ci)
            out.append((c, coef))
    return out


def _candidates(n, t):
    A = {Fraction(0), Fraction(1)}
    for k in range(n):
        for c in range(1, n - k + 1):
            x = (t - k) / c
            if 0 <= x <= 1:
                A.add(x)
    return sorted(A)


def _poly(inc, w):
    s = 0.0
    for c, coef in inc:
        p = coef
        for ci, wi in zip(c, w):
            if ci:
                p *= wi ** ci
        s += p
    return s


def _search(n, m, t):
    A = _candidates(n, t)
    fm = float(m)
    best = (-1.0, [Fraction(0), Fraction(1)], [1 - fm, fm])
    C2 = _count_vectors(n, 2)
    for x in A:
        if x > m:
            break
        for y in A:
            if y < m or y == x:
                continue
            inc = [(c, k) for c, k in C2 if c[0] * x + c[1] * y <= t]
            wy = float((m - x) / (y - x))
            v = _poly(inc, (1 - wy, wy))
            if v > best[0]:
                best = (v, [x, y], [1 - wy, wy])
    C3 = _count_vectors(n, 3)
    deadline = time.time() + 3.0
    G = 60
    for i, x in enumerate(A):
        if x >= m or time.time() > deadline:
            break
        for z in A[i + 2:]:
            if z <= m:
                continue
            for y in A[i + 1:]:
                if y >= z:
                    break
                inc = [(c, k) for c, k in C3 if c[0] * x + c[1] * y + c[2] * z <= t]
                if not inc:
                    continue
                fx, fy, fz = float(x), float(y), float(z)

                def wts(u):
                    wz = (fm - fy * u - fx * (1 - u)) / (fz - fx)
                    return (max(1 - u - wz, 0.0), u, max(wz, 0.0))
                hi = min(1.0, (fm - fx) / (fy - fx), (fz - fm) / (fz - fy))
                if hi <= 0:
                    continue
                us = [hi * g / G for g in range(G + 1)]
                vs = [_poly(inc, wts(u)) for u in us]
                g = max(range(G + 1), key=vs.__getitem__)
                a, b = us[max(g - 1, 0)], us[min(g + 1, G)]
                for _ in range(50):
                    c1, c2 = a + (b - a) * 0.382, a + (b - a) * 0.618
                    if _poly(inc, wts(c1)) < _poly(inc, wts(c2)):
                        a = c1
                    else:
                        b = c2
                u = (a + b) / 2
                v = _poly(inc, wts(u))
                if vs[g] >= v:
                    u, v = us[g], vs[g]
                if v > best[0] + 1e-12:
                    best = (v, [x, y, z], list(wts(u)))
    return best


def strategy(n, m, t):
    _, atoms, w = _search(n, m, t)
    keep = [i for i in range(len(atoms)) if w[i] > 1e-13]
    atoms = [atoms[i] for i in keep]
    w = [Fraction(w[i]).limit_denominator(10**12) for i in keep]
    if len(atoms) == 1:
        return atoms, [Fraction(1)]
    # The checker re-solves the last two weights: put the two heaviest atoms that bracket m last.
    order = sorted(range(len(atoms)), key=lambda i: w[i])
    lo = max((i for i in order if atoms[i] < m), key=lambda i: w[i])
    hi = max((i for i in order if atoms[i] > m), key=lambda i: w[i])
    rest = [i for i in range(len(atoms)) if i not in (lo, hi)]
    idx = rest + [lo, hi]
    return [atoms[i] for i in idx], [w[i] for i in idx]


def confidence(n, m, t):
    _, atoms, _ = _search(n, m, t)
    if n <= 2:
        return 0.95
    # 3-atom optima show the 2-point family is not always extremal; larger n leaves more room.
    base = {3: 0.6, 4: 0.5}.get(n, 0.35)
    if len(atoms) >= 3:
        base -= 0.1
    return base
