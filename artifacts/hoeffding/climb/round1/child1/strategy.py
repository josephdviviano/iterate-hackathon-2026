"""Structure: (fill in)
"""
import bisect
import math
import time
from fractions import Fraction


def _lcm(a, b):
    return a * b // math.gcd(a, b)


class _Law:
    """Atoms as integers in units of 1/D; sums of up to n atoms truncated at T = t D."""

    def __init__(self, n, m, t, D):
        self.n, self.m, self.D = n, float(m), D
        self.T = int(t * D)

    def cdf_n1(self, atoms, w):
        dist = {0: 1.0}
        T = self.T
        for _ in range(self.n - 1):
            nd = {}
            for s, p in dist.items():
                for a, wi in zip(atoms, w):
                    u = s + a
                    if u <= T:
                        nd[u] = nd.get(u, 0.0) + p * wi
            dist = nd
        keys = sorted(dist)
        cum, c = [], 0.0
        for k in keys:
            c += dist[k]
            cum.append(c)
        return keys, cum

    @staticmethod
    def at(keys, cum, x):
        i = bisect.bisect_right(keys, x)
        return cum[i - 1] if i else 0.0

    def grad(self, atoms, w):
        keys, cum = self.cdf_n1(atoms, w)
        g = [self.n * self.at(keys, cum, self.T - a) for a in atoms]
        return g, keys, cum


def _mean_tilt(w, z, af, m):
    """w_i exp(z_i - mu a_i), normalised, with mu chosen so the mean is m."""
    def make(mu):
        e = [math.log(wi) + zi - mu * (ai - m) if wi > 0 else -math.inf
             for wi, zi, ai in zip(w, z, af)]
        emax = max(e)
        e = [math.exp(x - emax) for x in e]
        s = sum(e)
        return [x / s for x in e]

    lo, hi = -1.0, 1.0
    while sum(x * a for x, a in zip(make(lo), af)) < m and lo > -1e6:
        lo *= 2
    while sum(x * a for x, a in zip(make(hi), af)) > m and hi < 1e6:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(x * a for x, a in zip(make(mid), af)) > m:
            lo = mid
        else:
            hi = mid
    return make((lo + hi) / 2)


def _optimize(L, atoms, w, deadline, iters=400):
    af = [a / L.D for a in atoms]
    g, _, _ = L.grad(atoms, w)
    v = sum(wi * gi for wi, gi in zip(w, g)) / L.n
    eta = 1.0
    for _ in range(iters):
        if time.monotonic() > deadline or eta < 1e-9:
            break
        w2 = _mean_tilt(w, [eta * gi for gi in g], af, L.m)
        g2, _, _ = L.grad(atoms, w2)
        v2 = sum(wi * gi for wi, gi in zip(w2, g2)) / L.n
        if v2 > v + 1e-15:
            w, g, v = w2, g2, v2
            eta *= 1.5
        else:
            eta /= 3
    return w, v


def _multipliers(atoms, w, g, D):
    """Least-squares fit g_i = lam + mu a_i over the support."""
    pts = [(a / D, gi) for a, wi, gi in zip(atoms, w, g) if wi > 1e-9]
    if len(pts) < 2:
        return pts[0][1], 0.0
    xs = sum(p[0] for p in pts) / len(pts)
    ys = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - xs) ** 2 for p in pts)
    mu = sum((p[0] - xs) * (p[1] - ys) for p in pts) / sxx if sxx else 0.0
    return ys - mu * xs, mu


def _column_generation(L, atoms, w, deadline, max_atoms=8):
    w, v = _optimize(L, atoms, w, deadline)
    kkt = False
    while time.monotonic() < deadline:
        keep = [i for i, x in enumerate(w) if x > 1e-10]
        atoms = [atoms[i] for i in keep]
        w = [w[i] for i in keep]
        s = sum(w)
        w = [x / s for x in w]
        g, keys, cum = L.grad(atoms, w)
        lam, mu = _multipliers(atoms, w, g, L.D)
        cand = {0, L.D}
        for k in keys:
            x = L.T - k
            if 0 <= x <= L.D:
                cand.add(x)
        cand -= set(atoms)
        scored = sorted(((L.n * L.at(keys, cum, L.T - x) - lam - mu * x / L.D, x) for x in cand),
                        reverse=True)
        if not scored or scored[0][0] < 1e-9 or len(atoms) >= max_atoms:
            kkt = not scored or scored[0][0] < 1e-9
            break
        improved = False
        for gap, x in scored[:3]:
            if gap < 1e-9:
                break
            a2 = atoms + [x]
            w2 = [wi * (1 - 1e-3) for wi in w] + [1e-3]
            w2, v2 = _optimize(L, a2, w2, deadline)
            if v2 > v + 1e-12:
                atoms, w, v, improved = a2, w2, v2, True
                break
        if not improved:
            break
    return atoms, w, v, kkt


def _starts(n, m, t):
    """Two-atom laws: k atoms at b and n - k at a with k b + (n - k) a = t."""
    out = []
    for k in range(n):
        if k > 0 and k * m >= t:
            break
        if k == 0:
            a, b = t / n, Fraction(1)
        else:
            a = max(Fraction(0), (t - k) / (n - k))
            b = (t - (n - k) * a) / k
        if not a < m < b <= 1:
            continue
        q = (m - a) / (b - a)
        p = sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k + 1))
        out.append((float(p), [a, b], [float(1 - q), float(q)]))
    out.sort(key=lambda c: -c[0])
    out.append((0.0, [Fraction(0), Fraction(1)], [float(1 - m), float(m)]))
    return out


def _search(n, m, t, budget=4.0):
    m, t = Fraction(m), Fraction(t)
    t0 = time.monotonic()
    starts = _starts(n, m, t)
    D = t.denominator
    for _, a, _ in starts:
        for x in a:
            D = _lcm(D, x.denominator)
    L = _Law(n, m, t, D)
    best = None
    for i, (_, a, w) in enumerate(starts):
        left = budget - (time.monotonic() - t0)
        if left <= 0:
            break
        deadline = time.monotonic() + left / max(1, min(len(starts) - i, 4))
        atoms = [int(x * D) for x in a]
        r = _column_generation(L, atoms, w, deadline)
        if best is None or r[2] > best[2]:
            best = r
    atoms, w, v, kkt = best
    return [Fraction(a, D) for a in atoms], w, v, kkt


def _order(atoms, w, m):
    """Put the heaviest atom below m and the heaviest above m last, for the mean repair."""
    lo = max((i for i in range(len(atoms)) if atoms[i] < m), key=lambda i: w[i])
    hi = max((i for i in range(len(atoms)) if atoms[i] > m), key=lambda i: w[i])
    idx = [i for i in range(len(atoms)) if i not in (lo, hi)] + [lo, hi]
    return [atoms[i] for i in idx], [Fraction(w[i]).limit_denominator(10 ** 12) for i in idx]


def strategy(n, m, t):
    atoms, w, v, kkt = _search(n, m, t)
    if len(atoms) == 1:
        return atoms, [1]
    return _order(atoms, w, Fraction(m))


def confidence(n, m, t):
    return 0.5
