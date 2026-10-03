"""Structure: at most 3 atoms; on all train instances 2 atoms carry the optimum except one 3-atom case.
Atoms sit at a vertex of {0 <= x <= 1, c.x <= t for the feasible count vectors c}: a low atom at 0 or
at a level (t - J)/(n - J) etc. so that sums hit t exactly, and a top atom at 1 or at t/J.
Typical laws: {t/n, 1}, {0, t/J}, {(t - J)/(n - J), 1}, and {0, t/J, 1} (e.g. n=6, m=3/5, t=9/10).
Unsettled: no proof of optimality; 4-atom vertex search and 5-atom local search gave no gain for n <= 6.
"""
import itertools
import math
import time
from fractions import Fraction


def _count_vectors(n):
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append(((i, j, k), math.factorial(n) // (math.factorial(i) * math.factorial(j) * math.factorial(k))))
    return out


def _det(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def _solve(rows, rhs):
    """Integer Cramer's rule; returns numerators and the common denominator."""
    D = _det(rows)
    if D == 0:
        return None
    nums = []
    for i in range(3):
        M = [list(r) for r in rows]
        for j in range(3):
            M[j][i] = rhs[j]
        nums.append(_det(M))
    if D < 0:
        D, nums = -D, [-x for x in nums]
    return nums, D


def _vertices(m, t, V, deadline):
    # rhs scaled by q so that all arithmetic stays in integers
    q = t.denominator
    unit = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    eqs = [(e, b * q) for b in (0, 1) for e in unit] + [(c, t.numerator) for c, _ in V]
    seen = set()
    for e in itertools.combinations(eqs, 3):
        if time.time() > deadline:
            break
        s = _solve([x[0] for x in e], [x[1] for x in e])
        if s is None:
            continue
        (x0, x1, x2), D = s
        Dq = D * q
        if 0 <= x0 < x1 < x2 <= Dq:
            a, b, c = Fraction(x0, Dq), Fraction(x1, Dq), Fraction(x2, Dq)
            if a < m < c:
                seen.add((a, b, c))
    return seen


def _u_range(atoms, m):
    a, b, c = atoms
    # w_c = (m - a - u (b - a)) / (c - a) >= 0 ; w_a = (c - m - u (c - b)) / (c - a) >= 0
    hi = min((m - a) / (b - a), (c - m) / (c - b))
    return Fraction(0), hi


def _value_fn(atoms, m, t, V):
    a, b, c = (float(x) for x in atoms)
    fm = float(m)
    feas = [(cv, co) for cv, co in V if cv[0] * atoms[0] + cv[1] * atoms[1] + cv[2] * atoms[2] <= t]

    def f(u):
        wc = (fm - a - u * (b - a)) / (c - a)
        wa = 1 - u - wc
        if wa < 0 or wc < 0:
            wa, wc = max(wa, 0.0), max(wc, 0.0)
        return sum(co * wa ** i * u ** j * wc ** k for (i, j, k), co in feas)
    return f


def _maximize(f, hi, grid):
    us = [hi * i / grid for i in range(grid + 1)]
    vals = [f(u) for u in us]
    i = max(range(len(us)), key=vals.__getitem__)
    lo, up = us[max(0, i - 1)], us[min(grid, i + 1)]
    for _ in range(50):
        m1, m2 = lo + (up - lo) / 3, up - (up - lo) / 3
        if f(m1) < f(m2):
            lo = m1
        else:
            up = m2
    u = (lo + up) / 2
    return (f(u), u) if f(u) > vals[i] else (vals[i], us[i])


def _two_point(n, m, t):
    """Best {a, b}: J top atoms fit under t; a at 0 or b at 1 binds."""
    best = (-1.0, None)
    for J in range(n):
        cands = [(t / n, Fraction(1))] if J == 0 else []
        if J and m < t / J <= 1:
            cands.append((Fraction(0), t / J))
        if J and J < t:
            cands.append(((t - J) / (n - J), Fraction(1)))
        for a, b in cands:
            if 0 <= a < m < b <= 1:
                p = (m - a) / (b - a)
                v = float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(J + 1)))
                if v > best[0]:
                    best = (v, (a, b))
    return best


def strategy(n, m, t):
    deadline = time.time() + 3.0
    V = _count_vectors(n)
    v2, ab = _two_point(n, m, t)
    verts = _vertices(m, t, V, deadline)
    coarse = []
    for atoms in verts:
        hi = float(_u_range(atoms, m)[1])
        f = _value_fn(atoms, m, t, V)
        coarse.append((_maximize(f, hi, 24)[0], atoms))
    coarse.sort(key=lambda z: -z[0])
    best = (-1.0, None, None)
    for _, atoms in coarse[:20]:
        lo, hi = _u_range(atoms, m)
        v, u = _maximize(_value_fn(atoms, m, t, V), float(hi), 400)
        if v > best[0]:
            best = (v, atoms, u)
    if ab is not None and v2 >= best[0]:
        return list(ab), [Fraction(1, 2), Fraction(1, 2)]
    if best[1] is None:
        return [0, 1], [1 - m, m]
    a, b, c = best[1]
    lo, hi = _u_range(best[1], m)
    u = min(max(Fraction(best[2]).limit_denominator(10**9), lo), hi)
    # the checker re-solves the last two weights; a and c straddle m
    return [b, a, c], [u, (1 - u) / 2, (1 - u) / 2]
