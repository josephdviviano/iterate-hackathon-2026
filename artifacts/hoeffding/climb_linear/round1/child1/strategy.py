"""Structure: (fill in)
"""
import random
import time
from bisect import bisect_left
from fractions import Fraction
from itertools import accumulate

TIME_BUDGET = 4.0
MAX_ATOMS = 64
WEIGHT_BITS = 30
DENSE_LIMIT = 64
VERIFY_OPS = 3e6
FINE = 1 << 62


def _convolve(dist, atoms, weights, coef):
    out = {}
    get = out.get
    for a, w in zip(atoms, weights):
        sh = coef * a
        for s, p in dist.items():
            k = s + sh
            out[k] = get(k, 0.0) + p * w
    return out


def _rests(c, atoms, weights):
    """For each distinct coefficient v: sorted sums and CDF of sum_{j != i} c_j X_j, c_i = v."""
    out = {}
    for v in set(c):
        rest = list(c)
        rest.remove(v)
        dist = {0: 1.0}
        for coef in rest:
            dist = _convolve(dist, atoms, weights, coef)
        keys = sorted(dist)
        cdf = [0.0] + list(accumulate(dist[k] for k in keys))
        out[v] = (keys, cdf, c.count(v))
    return out


def _grad(rests, points):
    g = []
    for a in points:
        t = 0.0
        for v, (keys, cdf, m) in rests.items():
            t += m * cdf[bisect_left(keys, -v * a)]
        g.append(t)
    return g


def _optimise(c, atoms, weights, cands, iters, deadline):
    n = len(c)
    gamma = 4.0
    rests = _rests(c, atoms, weights)
    grad = _grad(rests, atoms)
    val = sum(w * g for w, g in zip(weights, grad)) / n
    for _ in range(iters):
        if time.time() > deadline:
            break
        target = n * val
        new = [w * (g / target) ** gamma for w, g in zip(weights, grad)]
        new_atoms = list(atoms)
        have = set(atoms)
        pts = [a for a in cands if a not in have]
        if pts:
            gc = _grad(rests, pts)
            best = sorted((g, a) for g, a in zip(gc, pts) if g > target * 1.0005)
            for g, a in best[-3:]:
                new_atoms.append(a)
                new.append(0.02 / len(atoms))
        s = sum(new)
        order = sorted(range(len(new)), key=lambda i: -new[i])[:MAX_ATOMS]
        keep = [i for i in order if new[i] > 1e-7 * s]
        new_atoms = [new_atoms[i] for i in keep]
        s = sum(new[i] for i in keep)
        new = [new[i] / s for i in keep]
        r2 = _rests(c, new_atoms, new)
        g2 = _grad(r2, new_atoms)
        v2 = sum(w * g for w, g in zip(new, g2)) / n
        if v2 >= val:
            atoms, weights, rests, grad, val = new_atoms, new, r2, g2, v2
            gamma = min(gamma * 1.3, 64.0)
        else:
            gamma = max(gamma / 2, 0.5)
    return atoms, weights, val


def _verify_ops(c, atoms):
    sums = {0}
    ops = 0
    for coef in c:
        ops += len(sums) * len(atoms)
        sums = {s + coef * a for s in sums for a in atoms}
        if ops > 10 * VERIFY_OPS:
            break
    return ops


def _candidates(atoms, G):
    if G <= DENSE_LIMIT:
        return list(range(G + 1))
    pts = sorted(set(atoms) | {0, G})
    out = set()
    for lo, hi in zip(pts, pts[1:]):
        gap = hi - lo
        for num in (1, 2, 3):
            out.add(lo + gap * num // 4)
    return sorted(out)


def _search(c, budget):
    t0 = time.time()
    deadline = t0 + budget
    rng = random.Random(12345)
    G = 8
    pool = []
    for _ in range(12):
        w = [rng.random() ** 3 for _ in range(G + 1)]
        s = sum(w)
        w = [x / s for x in w]
        a = list(range(G + 1))
        a, w, v = _optimise(c, a, w, a, 40, t0 + budget * 0.1)
        pool.append((v, a, w))
    pool.sort(key=lambda t: -t[0])
    pool = pool[:3]
    while G < DENSE_LIMIT:
        G *= 2
        nxt = []
        for v, a, w in pool:
            a2 = [2 * x for x in a]
            nxt.append(_optimise(c, a2, w, _candidates(a2, G), 30, t0 + budget * 0.3))
        pool = sorted(((v, a, w) for a, w, v in nxt), key=lambda t: -t[0])
    up = FINE // G
    pool = [(v, [x * up for x in a], w) for v, a, w in pool]
    G = FINE
    best = pool[0]
    share = (deadline - time.time()) / len(pool)
    for v, a, w in pool:
        stop = time.time() + share
        while time.time() < stop:
            a2, w2, v2 = _optimise(c, a, w, _candidates(a, G), 5, stop)
            if _verify_ops(c, a2) > VERIFY_OPS:
                break
            a, w, v = a2, w2, v2
        if v > best[0]:
            best = (v, a, w)
    return best[0], G, best[1], best[2]


def _dyadic_weights(weights):
    scale = 1 << WEIGHT_BITS
    q = [max(0, round(w * scale)) for w in weights]
    q[q.index(max(q))] += scale - sum(q)
    return [Fraction(x, scale) for x in q]


_CACHE = {}


def _run(c):
    key = tuple(c)
    if key not in _CACHE:
        _CACHE[key] = _search(list(c), TIME_BUDGET)
    return _CACHE[key]


def strategy(c):
    val, G, atoms, weights = _run(c)
    pairs = sorted((a, w) for a, w in zip(atoms, weights) if w > 0)
    ws = _dyadic_weights([w for _, w in pairs])
    return [Fraction(a, G) for a, _ in pairs], ws


def confidence(c):
    return 0.0
