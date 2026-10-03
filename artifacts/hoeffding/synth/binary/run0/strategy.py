"""Structure: two atoms {a, b}, a < m < b, with k b + (n - k) a = t; for fixed k the
best choice is a = max(0, (t - k)/(n - k)), b = min(1, t/k) (k = 0: a = t/n, b = 1).
Sometimes a third atom does better, e.g. {0, t/k, 1} with mass on 1 to lift the mean.
Search: all 3-atom laws whose atoms are vertices of {c.a = t, a_i in {0, 1}}, weights by local search.
Open: 4-atom laws gave no gain on n <= 4 (full search) or n = 6 (partial search); optimality not proven.
"""
import itertools
import math
import random
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


def _solve(A, b):
    k = len(A)
    M = [[Fraction(x) for x in row] + [Fraction(bb)] for row, bb in zip(A, b)]
    for i in range(k):
        p = next((r for r in range(i, k) if M[r][i] != 0), None)
        if p is None:
            return None
        M[i], M[p] = M[p], M[i]
        for r in range(k):
            if r != i and M[r][i] != 0:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][k] / M[i][i] for i in range(k)]


def _prob(w, D):
    return sum(co * math.prod(wi ** ci for wi, ci in zip(w, c) if ci) for c, co in D)


def _complete(free, a, m):
    """Weights with the last two solved for total mass 1 and mean m."""
    W = sum(free)
    M = sum(x * y for x, y in zip(free, a))
    w2 = ((m - M) - a[-2] * (1 - W)) / (a[-1] - a[-2])
    w = list(free) + [1 - W - w2, w2]
    return None if min(w) < 0 else w


def _optimize_weights(a, m, D, rng, starts=4):
    k = len(a)
    best = (-1.0, None)
    for _ in range(starts):
        free = [rng.random() / k for _ in range(k - 2)]
        w = _complete(free, a, m)
        if w is None:
            free = [0.0] * (k - 2)
            w = _complete(free, a, m)
        v = _prob(w, D)
        step = 0.1
        while step > 1e-11:
            improved = False
            for i in range(k - 2):
                for d in (step, -step):
                    f2 = list(free)
                    f2[i] = max(0.0, f2[i] + d)
                    w2 = _complete(f2, a, m)
                    if w2 is not None:
                        v2 = _prob(w2, D)
                        if v2 > v:
                            free, w, v, improved = f2, w2, v2, True
            if not improved:
                step /= 2
        if v > best[0]:
            best = (v, free)
    return best


def _two_atom(n, m, t):
    best = None
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
        if best is None or p > best[0]:
            best = (float(p), [a, b], [1 - q, q])
    return best


def _three_atom(n, m, t, rng):
    K = 3
    CV = _count_vectors(n, K)
    cons = []
    for i in range(K):
        e = tuple(int(j == i) for j in range(K))
        cons += [(e, Fraction(1)), (e, Fraction(0))]
    cons += [(c, t) for c, _ in CV]
    deadline = time.monotonic() + 3.5
    seen = set()
    best = None
    mf = float(m)
    for S in itertools.combinations(cons, K):
        if time.monotonic() > deadline:
            break
        a = _solve([s[0] for s in S], [s[1] for s in S])
        if a is None or any(x < 0 or x > 1 for x in a):
            continue
        a = tuple(sorted(a))
        if len(set(a)) < K or a in seen or not a[0] < m < a[-1]:
            continue
        seen.add(a)
        # Extreme atoms last, so the mean repair acts on them.
        a = (a[1], a[0], a[2])
        D = [(c, co) for c, co in CV if sum(ci * ai for ci, ai in zip(c, a)) <= t]
        af = [float(x) for x in a]
        v, free = _optimize_weights(af, mf, D, rng)
        if best is None or v > best[0]:
            best = (v, list(a), free)
    if best is None:
        return None
    v, a, free = best
    free = [Fraction(x).limit_denominator(10 ** 12) for x in free]
    return v, a, _complete(free, a, m)


def strategy(n, m, t):
    m, t = Fraction(m), Fraction(t)
    rng = random.Random(0)
    cands = [c for c in (_two_atom(n, m, t), _three_atom(n, m, t, rng)) if c and c[2]]
    if not cands:
        return [0, 1], [1 - m, m]
    v, atoms, weights = max(cands, key=lambda c: c[0])
    return atoms, weights
