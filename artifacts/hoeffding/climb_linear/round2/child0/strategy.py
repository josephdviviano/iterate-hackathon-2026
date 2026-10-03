# Structure: atoms are cluster centres (0, 1/2, 1) plus signed perturbations s*delta*r^j on a
# shared ladder of levels j, with r < 1/(sum|c|+1), so a tie in sum c_i centre_i is broken by
# the sign of the coefficient sum at the shallowest perturbed level. All train optima use two
# clusters: 1 - r^j and 0 + r^j (0 + deep levels act as a tie-broken mass at 0). Unsettled: the
# finite-ladder loss O(1/levels) and whether richer (non-lexicographic) laws beat the limit.
import time
from fractions import Fraction
from math import comb, gcd

TIME_BUDGET = 3.5
MAX_ATOMS = 64
VERIFY_OPS = 1.5e6
WEIGHT_BITS = 40
D = 2


def _clusters():
    out = []
    for z in range(D + 1):
        for s in (1, -1):
            if (z == 0 and s < 0) or (z == D and s > 0):
                continue
            out.append((z, s))
    return out


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


def _grads(c, points, pert, L):
    """points[z]: weight of exact centre z; pert[(z, s)][j]: weight at level j.
    Returns value and gradients of the lexicographic model."""
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
    value = 0.0
    n = len(c)
    for v in set(c):
        rest = list(c)
        rest.remove(v)
        cnt = c.count(v)
        A = []
        for j in range(L):
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
                b = sum(p * _outcome(m + v * z, cs + v * s) for (m, cs), p in ra.items())
                gq[(z, s)][j] += cnt * b
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
    value = (sum(w * g for w, g in zip(points, gp))
             + sum(w * g for k in clusters for w, g in zip(pert[k], gq[k]))) / n
    return value, gp, gq


def _optimise(c, points, pert, L, deadline, iters):
    n = len(c)
    gamma = 2.0
    val, gp, gq = _grads(c, points, pert, L)
    for _ in range(iters):
        if time.time() > deadline:
            break
        t = n * val
        np_ = [w * (g / t) ** gamma for w, g in zip(points, gp)]
        nq = {k: [w * (g / t) ** gamma for w, g in zip(pert[k], gq[k])] for k in pert}
        tot = sum(np_) + sum(sum(x) for x in nq.values())
        np_ = [x / tot for x in np_]
        nq = {k: [x / tot if x / tot > 1e-12 else 0.0 for x in xs] for k, xs in nq.items()}
        v2, gp2, gq2 = _grads(c, np_, nq, L)
        if v2 >= val:
            points, pert, val, gp, gq = np_, nq, v2, gp2, gq2
            gamma = min(gamma * 1.5, 32.0)
        else:
            gamma = max(gamma / 2, 1.0)
    return val, points, pert


def _ops_estimate(c, K):
    """Upper bound on verification work: distinct partial sums are at most multisets per coefficient."""
    ops = 0
    seen = {}
    for coef in c:
        size = 1
        for m in seen.values():
            size *= comb(K + m - 1, m)
        ops += size * K
        seen[coef] = seen.get(coef, 0) + 1
    return ops


def _ops_exact(c, ints, limit):
    sums = {0}
    ops = 0
    for coef in c:
        ops += len(sums) * len(ints)
        if ops > limit:
            return ops
        sums = {x + coef * a for x in sums for a in ints}
    return ops


def _realise(c, points, pert, L):
    b = (sum(abs(x) for x in c) + 1).bit_length()
    atoms, weights = [], []
    for z in range(D + 1):
        if points[z] > 0:
            atoms.append(Fraction(z, D))
            weights.append(points[z])
    for (z, s), ws in pert.items():
        for j, w in enumerate(ws):
            if w > 0:
                atoms.append(Fraction(z, D) + s * Fraction(1, D * 2 ** (b * (j + 1))))
                weights.append(w)
    return atoms, weights


def _count(points, pert):
    return sum(1 for x in points if x > 0) + sum(1 for xs in pert.values() for x in xs if x > 0)


def _prune(points, pert, keep):
    items = [(w, ('p', z)) for z, w in enumerate(points) if w > 0]
    items += [(w, (k, j)) for k, xs in pert.items() for j, w in enumerate(xs) if w > 0]
    items.sort(reverse=True)
    points = [0.0] * (D + 1)
    pert = {k: [0.0] * len(xs) for k, xs in pert.items()}
    for w, (k, j) in items[:keep]:
        if k == 'p':
            points[j] = w
        else:
            pert[k][j] = w
    tot = sum(points) + sum(sum(x) for x in pert.values())
    return [x / tot for x in points], {k: [x / tot for x in xs] for k, xs in pert.items()}


def _normalise(points, pert):
    tot = sum(points) + sum(sum(x) for x in pert.values())
    return [x / tot for x in points], {k: [x / tot for x in xs] for k, xs in pert.items()}


def _fits(c, points, pert, L):
    atoms, _ = _realise(c, points, pert, L)
    if len(atoms) > MAX_ATOMS:
        return False
    den = 1
    for a in atoms:
        den = den * a.denominator // gcd(den, a.denominator)
    return _ops_exact(c, [int(a * den) for a in atoms], VERIFY_OPS) <= VERIFY_OPS


def _search(c):
    t0 = time.time()
    clusters = _clusters()
    # Coarse ladder to find the active clusters.
    L0 = 8
    points, pert = _normalise([1.0] * (D + 1), {k: [1.0] * L0 for k in clusters})
    val, points, pert = _optimise(c, points, pert, L0, t0 + TIME_BUDGET * 0.15, 200)
    mass = {k: sum(xs) for k, xs in pert.items()}
    ranked = sorted((k for k in clusters if mass[k] > 0.01), key=lambda k: -mass[k])
    pts = [points[z] if points[z] > 0.01 else 0.0 for z in range(D + 1)]
    npts = sum(1 for x in pts if x > 0)
    # Each subset of the heaviest clusters shares the atom budget; profiles start
    # uniform per cluster and as a blend with the coarse profile.
    tasks = []
    for size in range(1, min(3, len(ranked)) + 1):
        active = ranked[:size]
        hi = (MAX_ATOMS - npts) // size
        lo = 1
        while lo < hi and _ops_estimate(c, npts + size * (lo + 1)) <= VERIFY_OPS:
            lo += 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _fits(c, pts, {k: [1.0] * mid for k in active}, mid):
                lo = mid
            else:
                hi = mid - 1
        L = lo
        uniform = {k: [mass[k] / L] * L for k in active}
        stretched = {k: [pert[k][min(L0 - 1, j * L0 // L)] * L0 / L for j in range(L)] for k in active}
        blend = {k: [(a + b) / 2 for a, b in zip(stretched[k], uniform[k])] for k in active}
        tasks += [(L, uniform), (L, blend)]
    best = None
    start = time.time()
    share = (t0 + TIME_BUDGET - start) / len(tasks)
    for i, (L, q) in enumerate(tasks):
        p0, q0 = _normalise(pts, q)
        res = _optimise(c, p0, q0, L, start + share * (i + 1), 10 ** 6)
        if best is None or res[0] > best[0]:
            best = res + (L,)
    val, np_, nq, L = best
    atoms, weights = _realise(c, np_, nq, L)
    return val, atoms, weights


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
    val, atoms, weights = _run(c)
    return atoms, _dyadic(weights)


def confidence(c):
    return 0.0
