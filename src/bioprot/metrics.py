"""Risk-coverage statistics as pure functions of (risk, uncertainty) pairs.

Items are sorted by uncertainty ascending, ties broken by a deterministic key,
and the abstention threshold sweeps from full coverage down to zero. The
random baseline is flat at the full-coverage risk; the oracle sorts by the
true risk. Bootstrap intervals resample groups (protocols), not items.
"""

from __future__ import annotations

import numpy as np


def _order(unc, tiebreak):
    unc = np.asarray(unc, dtype=float)
    tb = np.asarray(tiebreak)
    return np.lexsort((tb, unc))


def risk_coverage_curve(risk, unc, tiebreak) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and the mean risk of the k least uncertain items, for k = 1..n."""
    r = np.asarray(risk, dtype=float)[_order(unc, tiebreak)]
    k = np.arange(1, len(r) + 1)
    return k / len(r), np.cumsum(r) / k


def aurc(risk, unc, tiebreak) -> float:
    return float(risk_coverage_curve(risk, unc, tiebreak)[1].mean())


def selective_risk_at(risk, unc, tiebreak, coverage: float) -> float:
    cov, sel = risk_coverage_curve(risk, unc, tiebreak)
    k = max(1, int(np.ceil(coverage * len(cov) - 1e-9)))
    return float(sel[k - 1])


def coverage_at_risk(risk, unc, tiebreak, target: float) -> float:
    """The largest coverage whose selective risk is at or below the target."""
    cov, sel = risk_coverage_curve(risk, unc, tiebreak)
    ok = np.nonzero(sel <= target)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def random_aurc(risk) -> float:
    return float(np.mean(risk))


def oracle_aurc(risk, tiebreak) -> float:
    return aurc(risk, risk, tiebreak)


def tie_fraction(unc) -> float:
    """Share of items whose uncertainty value is shared with at least one other item."""
    u = np.asarray(unc)
    _, inv, counts = np.unique(u, return_inverse=True, return_counts=True)
    return float(np.mean(counts[inv] > 1))


def bootstrap(groups, stat, n_boot: int = 1000, seed: int = 0):
    """95% interval of stat(index_array), scalar or vector, over resampled groups with replacement."""
    g = np.asarray(groups)
    ids = np.unique(g)
    members = {i: np.nonzero(g == i)[0] for i in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[i] for i in draw])
        vals.append(stat(idx))
    lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5], axis=0)
    return (float(lo), float(hi)) if np.ndim(lo) == 0 else (lo, hi)


def ece(conf, correct, n_bins: int = 10) -> float:
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.any():
            out += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(out)


def brier(conf, correct) -> float:
    return float(np.mean((np.asarray(conf, dtype=float) - np.asarray(correct, dtype=float)) ** 2))
