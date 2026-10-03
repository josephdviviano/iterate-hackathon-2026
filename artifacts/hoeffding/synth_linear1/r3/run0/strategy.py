# Structure: atoms are centres z/2 (z = 0, 1, 2) plus signed perturbations s*r^j on a ladder of
# levels, r < 1/(sum|c|+1), so ties in sum c_i z_i are broken lexicographically. Mass at centre 0
# is a scaled copy of the whole law (self-similar recursion). A limit model (infinite ladders,
# cluster depth ranks, copy fixed point) picks the structure; finite ladders are then optimised.
# Unsettled: whether the limit model's optimum is the true supremum, and the finite-ladder loss.
import itertools
import time
from fractions import Fraction
from math import comb, factorial, gcd, inf

TIME_BUDGET = 3.5
MAX_ATOMS = 64
VERIFY_OPS = 1.5e6
WEIGHT_BITS = 40
D = 2
LADDER = [(0, 1), (1, 1), (1, -1), (2, -1)]


# Limit model: each cluster is an infinite continuous ladder; clusters of equal rank share levels.

def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for x in range(total + 1):
        for rest in _compositions(total - x, parts - 1):
            yield (x,) + rest


def _limit_terms(c, clusters):
    groups = sorted({v: c.count(v) for v in c}.items())
    k = len(clusters)
    terms = []
    for combo in itertools.product(*[list(_compositions(m, k)) for _, m in groups]):
        coef = 1
        exps = [0] * k
        main = 0
        for (v, m), parts in zip(groups, combo):
            coef *= factorial(m)
            for a, x in enumerate(parts):
                coef //= factorial(x)
                exps[a] += x
                main += v * x * clusters[a][0]
        if main > 0:
            continue
        if main < 0:
            terms.append((coef, exps, 1.0))
            continue
        rank = min(clusters[a][2] for a in range(k) if exps[a])
        if rank == inf:
            terms.append((coef, exps, None))
            continue
        tot = sum(exps[a] for a in range(k) if clusters[a][2] == rank)
        neg = sum(x for (v, _), parts in zip(groups, combo) for a, x in enumerate(parts)
                  if clusters[a][2] == rank and v * clusters[a][1] < 0)
        terms.append((coef, exps, neg / tot))
    return terms


def _limit_value(terms, pi):
    a = 0.0
    tie = 0.0
    for coef, exps, out in terms:
        p = coef
        for w, e in zip(pi, exps):
            if e:
                p *= w ** e
        if out is None:
            tie += p
        else:
            a += p * out
    return a / (1 - tie) if tie < 1 else 0.0


def _limit_opt(terms, k):
    best = (-1.0, None)
    for start in range(k):
        pi = [0.5 / k] * k
        pi[start] += 0.5
        v = _limit_value(terms, pi)
        step = 0.2
        while step > 1e-4:
            moved = False
            for a in range(k):
                for b in range(k):
                    if a == b or pi[b] < step:
                        continue
                    q = list(pi)
                    q[a] += step
                    q[b] -= step
                    w = _limit_value(terms, q)
                    if w > v + 1e-12:
                        v, pi, moved = w, q, True
            if not moved:
                step /= 2
        if v > best[0]:
            best = (v, pi)
    return best


def _structures():
    out = []
    for m in (1, 2):
        for cl in itertools.combinations(LADDER, m):
            for ranks in itertools.product(range(m), repeat=m):
                if min(ranks) != 0:
                    continue
                base = [(z, s, r) for (z, s), r in zip(cl, ranks)]
                out.append(base)
                if cl[0] != (0, 1):
                    out.append(base + [(0, 1, inf)])
    return out


def _limit_search(c):
    best = None
    for clusters in _structures():
        terms = _limit_terms(c, clusters)
        v, pi = _limit_opt(terms, len(clusters))
        if best is None or v > best[0] + 1e-9:
            best = (v, clusters, pi)
    return best


# Finite model: lexicographic value of a law on a shared ladder, exact centre points, and a copy
# of value vcopy standing in for the point at centre 0.

def _outcome(main, cs):
    return 1.0 if main < 0 or (main == 0 and cs < 0) else 0.0


def _conv(dist, coef, opts):
    out = {}
    get = out.get
    for (m, cs), p in dist.items():
        for dz, ds, q in opts:
            k = (m + coef * dz, cs + coef * ds)
            out[k] = get(k, 0.0) + p * q
    return out


def _grads(c, points, pert, L, vcopy):
    clusters = list(pert)
    gp = [0.0] * (D + 1)
    gq = {k: [0.0] * L for k in clusters}
    deep = list(points)
    tails = []
    for j in range(L - 1, -1, -1):
        tails.append(list(deep))
        for (z, s) in clusters:
            deep[z] += pert[(z, s)][j]
    tails.reverse()
    n = len(c)
    for v in set(c):
        rest = list(c)
        rest.remove(v)
        cnt = c.count(v)
        A = []
        for j in range(L):
            if not any(pert[k][j] > 0 for k in clusters):
                A.append([0.0] * (D + 1))
                continue
            dj = tails[j]
            dopts = [(z, 0, dj[z]) for z in range(D + 1) if dj[z] > 0]
            aopts = dopts + [(z, s, pert[(z, s)][j]) for (z, s) in clusters if pert[(z, s)][j] > 0]
            ra = {(0, 0): 1.0}
            rd = {(0, 0): 1.0}
            for coef in rest:
                ra = _conv(ra, coef, aopts)
                rd = _conv(rd, coef, dopts)
            Aj = []
            for z in range(D + 1):
                t = sum(p * _outcome(m + v * z, cs) for (m, cs), p in ra.items())
                t -= sum(p for (m, _), p in rd.items() if m + v * z < 0)
                Aj.append(t)
            A.append(Aj)
            for (z, s) in clusters:
                if pert[(z, s)][j] > 0:
                    gq[(z, s)][j] += cnt * sum(p * _outcome(m + v * z, cs + v * s)
                                               for (m, cs), p in ra.items())
        dopts = [(z, 0, points[z]) for z in range(D + 1) if points[z] > 0]
        rp = {(0, 0): 1.0}
        for coef in rest:
            rp = _conv(rp, coef, dopts)
        pre = [0.0] * (D + 1)
        for j in range(L):
            for (z, s) in clusters:
                gq[(z, s)][j] += cnt * pre[z]
            for z in range(D + 1):
                pre[z] += A[j][z]
        for z in range(D + 1):
            tinf = sum(p for (m, _), p in rp.items() if m + v * z < 0)
            gp[z] += cnt * (pre[z] + tinf)
        gp[0] += cnt * points[0] ** (n - 1) * vcopy
    value = (sum(w * g for w, g in zip(points, gp))
             + sum(w * g for k in clusters for w, g in zip(pert[k], gq[k]))) / n
    return value, gp, gq


def _optimise(c, points, pert, L, vcopy, deadline):
    n = len(c)
    gamma = 2.0
    val, gp, gq = _grads(c, points, pert, L, vcopy)
    while time.time() < deadline:
        t = n * val
        np_ = [w * (g / t) ** gamma for w, g in zip(points, gp)]
        nq = {k: [w * (g / t) ** gamma for w, g in zip(pert[k], gq[k])] for k in pert}
        tot = sum(np_) + sum(sum(x) for x in nq.values())
        np_ = [x / tot for x in np_]
        nq = {k: [x / tot for x in xs] for k, xs in nq.items()}
        v2, gp2, gq2 = _grads(c, np_, nq, L, vcopy)
        if v2 >= val:
            done = v2 - val < 1e-10 and gamma >= 32.0
            points, pert, val, gp, gq = np_, nq, v2, gp2, gq2
            gamma = min(gamma * 1.5, 32.0)
            if done:
                break
        else:
            gamma = gamma / 2
            if gamma < 0.05:
                break
    return val, points, pert


def _layout(clusters, pi, L):
    """Initial weights: each rank gets a window of levels sized by its mass."""
    ladder = [(z, s, r, w) for (z, s, r), w in zip(clusters, pi) if r != inf and w > 1e-6]
    ranks = sorted({r for _, _, r, _ in ladder})
    mass = {r: sum(w for _, _, q, w in ladder if q == r) for r in ranks}
    size = {r: sum(1 for _, _, q, _ in ladder if q == r) for r in ranks}
    total = sum(mass[r] * size[r] for r in ranks)
    windows = {}
    start = 0
    for i, r in enumerate(ranks):
        width = L - start if i == len(ranks) - 1 else max(1, round(L * mass[r] * size[r] / total))
        windows[r] = (start, start + width)
        start += width
    points = [0.0] * (D + 1)
    pert = {}
    for z, s, r, w in ladder:
        lo, hi = windows[r]
        levels = hi - lo
        pert[(z, s)] = [0.0] * L
        for j in range(lo, hi):
            pert[(z, s)][j] = 0.9 * w / (levels + 1)
        points[z] += 0.1 * w + 0.9 * w / (levels + 1)
    for (z, s, r), w in zip(clusters, pi):
        if r == inf:
            points[0] += w
    tot = sum(points) + sum(map(sum, pert.values()))
    return [x / tot for x in points], {k: [x / tot for x in xs] for k, xs in pert.items()}


def _levels(clusters, pi, K):
    """Ladder length so that the top law has at most K atoms besides the copy."""
    ladder = [(z, s, r) for (z, s, r), w in zip(clusters, pi) if r != inf and w > 1e-6]
    per_rank = {}
    for _, _, r in ladder:
        per_rank[r] = per_rank.get(r, 0) + 1
    width = max(per_rank.values())
    npts = len({z for z, _, _ in ladder} | ({0} if any(r == inf for _, _, r in clusters) else set()))
    return max(1, (K - npts) // width)


def _bits(c):
    return (sum(abs(x) for x in c) + 1).bit_length()


def _realise(c, points, pert, L, copy):
    b = _bits(c)
    atoms, weights = [], []
    for z in range(D + 1):
        if points[z] <= 0:
            continue
        if z == 0 and copy is not None:
            scale = Fraction(1, D * 2 ** (b * (L + 1)))
            for a, w in zip(*copy):
                atoms.append(a * scale)
                weights.append(w * points[0])
        else:
            atoms.append(Fraction(z, D))
            weights.append(points[z])
    for (z, s), ws in pert.items():
        for j, w in enumerate(ws):
            if w > 0:
                atoms.append(Fraction(z, D) + s * Fraction(1, D * 2 ** (b * (j + 1))))
                weights.append(w)
    return atoms, weights


def _ops_exact(c, atoms, limit):
    den = 1
    for a in atoms:
        den = den * a.denominator // gcd(den, a.denominator)
    ints = [int(a * den) for a in atoms]
    sums = {0}
    ops = 0
    for coef in c:
        ops += len(sums) * len(ints)
        if ops > limit:
            return ops
        sums = {x + coef * a for x in sums for a in ints}
    return ops


def _ops_estimate(c, K):
    ops = 0
    seen = {}
    for coef in c:
        size = 1
        for m in seen.values():
            size *= comb(K + m - 1, m)
        ops += size * K
        seen[coef] = seen.get(coef, 0) + 1
    return ops


def _prune(atoms, weights, eps=1e-9):
    keep = [(a, w) for a, w in zip(atoms, weights) if w > eps]
    tot = sum(w for _, w in keep)
    return [a for a, _ in keep], [w / tot for _, w in keep]


def _build(c, structure, K, copy, deadline):
    """Optimise a top law of at most K atoms around a copy law (atoms, weights, value)."""
    v, clusters, pi = structure
    L = _levels(clusters, pi, K)
    points, pert = _layout(clusters, pi, L)
    if copy is None:
        vcopy = 0.0
    else:
        vcopy = copy[2]
    val, points, pert = _optimise(c, points, pert, L, vcopy, deadline)
    atoms, weights = _realise(c, points, pert, L, None if copy is None else copy[:2])
    atoms, weights = _prune(atoms, weights)
    return atoms, weights, val


TAILS = [(), (4,), (6,), (8,), (12,), (5, 3), (6, 4), (8, 4), (8, 5, 3), (10, 6, 4)]


def _significant(c, structure):
    """Drop clusters of negligible limit mass and re-optimise the rest."""
    _, clusters, pi = structure
    kept = [cl for cl, w in zip(clusters, pi) if w > 0.02]
    terms = _limit_terms(c, kept)
    v, pi = _limit_opt(terms, len(kept))
    return v, kept, pi


def _fits(c, atoms):
    return len(atoms) <= MAX_ATOMS and _ops_exact(c, atoms, VERIFY_OPS) <= VERIFY_OPS


def _search(c):
    t0 = time.time()
    structure = _significant(c, _limit_search(c))
    _, clusters, pi = structure
    has_copy = any(r == inf for _, _, r in clusters)
    budget = MAX_ATOMS
    while budget > 2 and _ops_estimate(c, budget) > VERIFY_OPS:
        budget -= 1
    tails = [t for t in TAILS if sum(t) <= budget // 2] if has_copy else [()]
    copies = {(): None}

    def copy_of(t):
        if t not in copies:
            copies[t] = _build(c, structure, t[0], copy_of(t[1:]), time.time() + 0.05)
        return copies[t]

    for t in tails:
        copy_of(t)
    share = (t0 + TIME_BUDGET - time.time()) / len(tails)
    best = None
    for t in tails:
        copy = copies[t]
        K = budget + 1 - (len(copy[0]) if copy else 0)
        deadline = time.time() + share
        while True:
            res = _build(c, structure, K, copy, deadline)
            if _fits(c, res[0]) or K <= 2:
                break
            K -= 1
        if _fits(c, res[0]) and (best is None or res[2] > best[2]):
            best = res
    return best[2], best[0], best[1], structure


def _dyadic(weights):
    scale = 1 << WEIGHT_BITS
    q = [max(0, round(w * scale)) for w in weights]
    q[q.index(max(q))] += scale - sum(q)
    return [Fraction(x, scale) for x in q]


_CACHE = {}


def _run(c):
    key = tuple(c)
    if key not in _CACHE:
        _CACHE[key] = _search(list(c))
    return _CACHE[key]


def strategy(c):
    val, atoms, weights, _ = _run(c)
    return atoms, _dyadic(weights)


def confidence(c):
    return 0.0
