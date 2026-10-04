"""Greedy Polar-Express-style composition (arXiv 2505.16932, reimplemented): each step is the minimax odd quintic
p(x) = a x + b x^3 + c x^5 approximating 1 on [l, u] (Remez on the 4-point alternation set {l, x1, x2, u}, where x1, x2
are p's critical points), then [l, u] <- [min p, max p] on the old interval. Safety: coefficients of every step are
scaled by 1/1.01 in the polynomial's input (a/1.01, b/1.01^3, c/1.01^5), as in the paper's code."""
import numpy as np


def minimax_quintic(l, u, iters=100):
    xs = np.array([l, l + (u - l) * 0.2, l + (u - l) * 0.6, u])
    for _ in range(iters):
        A = np.stack([xs, xs**3, xs**5, (-1.0) ** np.arange(4)], axis=1)
        a, b, c, E = np.linalg.solve(A, np.ones(4))
        disc = (3 * b) ** 2 - 4 * 5 * c * a  # roots of a + 3b y + 5c y^2 = 0, y = x^2
        if disc < 0:
            break
        ys = sorted([(-3 * b - np.sqrt(disc)) / (10 * c), (-3 * b + np.sqrt(disc)) / (10 * c)])
        crit = [np.sqrt(y) for y in ys if y > 0]
        crit = [x for x in crit if l < x < u]
        if len(crit) != 2:
            break
        new = np.array([l, crit[0], crit[1], u])
        if np.allclose(new, xs, rtol=1e-12):
            break
        xs = new
    return a, b, c


def optimal_composition(num_iters, l, u=1.0, safety=1.01):
    coeffs = []
    for _ in range(num_iters):
        a, b, c = minimax_quintic(l, u)
        grid = np.concatenate([np.geomspace(l, u, 20000)])
        p = a * grid + b * grid**3 + c * grid**5
        l, u = p.min(), p.max()
        coeffs.append((a / safety, b / safety**3, c / safety**5))
    return coeffs, (l, u)


if __name__ == "__main__":
    import sys
    l = float(sys.argv[1]) if len(sys.argv) > 1 else 7.86e-3
    coeffs, (lo, hi) = optimal_composition(3, l)
    for c in coeffs:
        print("(%.6f, %.6f, %.6f)" % c)
    print("final interval [%.4f, %.4f]" % (lo, hi))
    # compare with the fixed (3.4445, -4.7750, 2.0315) x3 on the same grid
    x = np.geomspace(l, 1, 2000)
    f = lambda x, abc: abc[0] * x + abc[1] * x**3 + abc[2] * x**5
    y_pe, y_ns = x.copy(), x.copy()
    for c in coeffs:
        y_pe = f(y_pe, c)
    for _ in range(3):
        y_ns = f(y_ns, (3.4445, -4.7750, 2.0315))
    print("on [l,1]: PE min %.3f max %.3f | NS3 min %.3f max %.3f" % (y_pe.min(), y_pe.max(), y_ns.min(), y_ns.max()))
