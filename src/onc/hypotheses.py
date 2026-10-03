"""Hypothesis programs for ONC-AGI worlds, one template per causal role family.

A hypothesis reads the revealed baseline data and claims an ordered driver set
and a model p(y | x). Post-outcome columns never reach a template: the leak
filter is applied to the input, not learned. Every template selects its
features inside the fit, so cross-validated log loss covers the selection.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

Array = NDArray[np.float64]

FDR = 0.05
MAX_DRIVERS = 4
MIN_ROWS = 12
CV_FOLDS = 5


@dataclass(frozen=True)
class Features:
    """Baseline columns only, standardised, with their metadata."""

    ids: tuple[str, ...]
    types: tuple[str, ...]
    x: Array
    y: NDArray[np.int64]
    stratum: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def column(self, fid: str) -> int:
        return self.ids.index(fid)


def features_from(card, feature_ids, x, y, stratum) -> Features:
    """Drop post-outcome and unmeasured columns, impute by median, standardise."""
    timing = {f.feature_id: f.timing.value for f in card.features}
    dtype = {f.feature_id: f.data_type for f in card.features}
    keep = [j for j, f in enumerate(feature_ids) if timing[f] == "baseline" and not np.isnan(x[:, j]).all()]
    cols = x[:, keep].astype(float)
    if cols.size:
        cols = np.where(np.isnan(cols), np.nanmedian(cols, axis=0), cols)
        cols = np.nan_to_num(cols)
        sd = cols.std(axis=0)
        cols = (cols - cols.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    ids = tuple(feature_ids[j] for j in keep)
    return Features(ids, tuple(dtype[f] for f in ids), cols, np.asarray(y, dtype=np.int64), tuple(stratum))


# ---------------------------------------------------------------------- designs


@dataclass(frozen=True)
class Design:
    """How a hypothesis builds its model input from the feature matrix."""

    drivers: tuple[str, ...]
    columns: tuple[str, ...] = ()
    products: tuple[tuple[str, str], ...] = ()
    loadings: dict[str, float] = field(default_factory=dict)

    def matrix(self, feats: Features) -> Array:
        parts: list[Array] = []
        for fid in self.columns:
            parts.append(feats.x[:, feats.column(fid)])
        for a, b in self.products:
            parts.append(feats.x[:, feats.column(a)] * feats.x[:, feats.column(b)])
        if self.loadings:
            score = sum(w * feats.x[:, feats.column(f)] for f, w in self.loadings.items())
            parts.append(np.asarray(score))
        if not parts:
            return np.empty((feats.n, 0))
        return np.column_stack(parts)


Template = Callable[[Features], Design]


@dataclass
class Hypothesis:
    name: str
    design: Design
    coef: Array
    intercept: float
    cv_logloss: float

    @property
    def drivers(self) -> tuple[str, ...]:
        return self.design.drivers

    @property
    def is_null(self) -> bool:
        return not self.design.drivers

    def predict(self, feats: Features) -> Array:
        z = self.intercept + self.design.matrix(feats) @ self.coef
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------- screens


def _bh(p: Array, fdr: float = FDR) -> list[int]:
    """Indices that pass Benjamini-Hochberg, ordered by p-value."""
    if p.size == 0:
        return []
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= fdr * np.arange(1, m + 1) / m
    k = int(np.max(np.flatnonzero(passed)) + 1) if passed.any() else 0
    return [int(j) for j in order[:k]]


def _marginal_p(feats: Features, columns: Array | None = None) -> Array:
    """Welch t-test p-value per column against the outcome."""
    x = feats.x if columns is None else columns
    y = feats.y.astype(bool)
    if x.shape[1] == 0:
        return np.empty(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = stats.ttest_ind(x[y], x[~y], equal_var=False).pvalue
    return np.nan_to_num(np.atleast_1d(p), nan=1.0)


def _adjusted_p(feats: Features, covariates: Array) -> Array:
    """P-value of each column after linear adjustment for the covariates (Frisch-Waugh)."""
    z = np.column_stack([np.ones(feats.n), covariates])
    proj = z @ np.linalg.pinv(z)
    rx = feats.x - proj @ feats.x
    ry = feats.y - proj @ feats.y.astype(float)
    dof = max(feats.n - z.shape[1] - 1, 1)
    sxx = (rx**2).sum(axis=0)
    beta = (rx * ry[:, None]).sum(axis=0) / np.where(sxx == 0, 1.0, sxx)
    resid = ry[:, None] - rx * beta
    se = np.sqrt((resid**2).sum(axis=0) / dof / np.where(sxx == 0, 1.0, sxx))
    t = np.where(se == 0, 0.0, beta / np.where(se == 0, 1.0, se))
    return 2 * stats.t.sf(np.abs(t), dof)


def _usable(feats: Features) -> bool:
    return feats.n >= MIN_ROWS and feats.x.shape[1] > 0 and 0 < feats.y.sum() < feats.n


# ---------------------------------------------------------------------- templates


def null(feats: Features) -> Design:
    return Design(drivers=())


def direct(feats: Features) -> Design:
    """Marginal screen: features that pass BH, strongest first."""
    if not _usable(feats):
        return Design(())
    hits = _bh(_marginal_p(feats))[:MAX_DRIVERS]
    ids = tuple(feats.ids[j] for j in hits)
    return Design(drivers=ids, columns=ids)


def conservative(feats: Features) -> Design:
    """Bonferroni screen: fewer claims, abstains more often."""
    if not _usable(feats):
        return Design(())
    p = _marginal_p(feats)
    hits = [int(j) for j in np.argsort(p) if p[j] < FDR / len(p)][:MAX_DRIVERS]
    ids = tuple(feats.ids[j] for j in hits)
    return Design(drivers=ids, columns=ids)


def sparse(feats: Features) -> Design:
    """L1 logistic regression: the non-zero coefficients, largest first."""
    if not _usable(feats):
        return Design(())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = LogisticRegression(l1_ratio=1.0, C=0.1, solver="liblinear").fit(feats.x, feats.y)
    coef = np.abs(fit.coef_[0])
    order = [int(j) for j in np.argsort(-coef) if coef[j] > 1e-8][:MAX_DRIVERS]
    ids = tuple(feats.ids[j] for j in order)
    return Design(drivers=ids, columns=ids)


def confounder(feats: Features) -> Design:
    """Clinical indicators and strata as covariates; a feature counts only if it survives adjustment."""
    if not _usable(feats):
        return Design(())
    clinical = [j for j, t in enumerate(feats.types) if t == "clinical"]
    strata = sorted(set(feats.stratum))
    dummies = [np.array([s == name for s in feats.stratum], dtype=float) for name in strata[1:]]
    covariates = np.column_stack([feats.x[:, clinical]] + dummies) if clinical or dummies else np.empty((feats.n, 0))
    if covariates.shape[1] == 0:
        return Design(())
    p_marg = _marginal_p(feats)
    clinical_hits = [j for j in _bh(p_marg) if j in clinical]
    p_adj = _adjusted_p(feats, covariates)
    others = [j for j in _bh(p_adj) if j not in clinical]
    hits = (clinical_hits + others)[:MAX_DRIVERS]
    ids = tuple(feats.ids[j] for j in hits)
    return Design(drivers=ids, columns=ids)


def upstream(feats: Features) -> Design:
    """Replace an expression hit by the copy-number feature that explains it, when that feature predicts as well."""
    if not _usable(feats):
        return Design(())
    p = _marginal_p(feats)
    hits = _bh(p)[:MAX_DRIVERS]
    if not hits:
        return Design(())
    corr = np.corrcoef(feats.x, rowvar=False) if feats.x.shape[1] > 1 else np.ones((1, 1))
    copy_number = [j for j, t in enumerate(feats.types) if t == "copy_number"]
    chosen: list[int] = []
    for j in hits:
        pick = j
        if feats.types[j] == "expression" and copy_number:
            linked = [c for c in copy_number if abs(corr[c, j]) >= 0.4]
            if linked:
                c = min(linked, key=lambda c: p[c])
                if p[c] <= np.sqrt(p[j]):
                    pick = c
        if pick not in chosen:
            chosen.append(pick)
    ids = tuple(feats.ids[j] for j in chosen)
    return Design(drivers=ids, columns=ids)


def interaction(feats: Features) -> Design:
    """Pairwise products adjusted for both main effects; the pairs that pass BH."""
    if not _usable(feats) or feats.x.shape[1] < 2:
        return Design(())
    p_cols = feats.x.shape[1]
    pairs = [(a, b) for a in range(p_cols) for b in range(a + 1, p_cols)]
    pvals = np.empty(len(pairs))
    y = feats.y.astype(float)
    for k, (a, b) in enumerate(pairs):
        z = np.column_stack([np.ones(feats.n), feats.x[:, a], feats.x[:, b]])
        prod = feats.x[:, a] * feats.x[:, b]
        proj = z @ np.linalg.pinv(z)
        rp, ry = prod - proj @ prod, y - proj @ y
        sxx = float((rp**2).sum())
        if sxx == 0:
            pvals[k] = 1.0
            continue
        beta = float((rp * ry).sum() / sxx)
        dof = feats.n - 4
        se = np.sqrt(((ry - rp * beta) ** 2).sum() / dof / sxx)
        pvals[k] = 2 * stats.t.sf(abs(beta / se), dof) if se > 0 else 1.0
    hits = _bh(pvals)[:3]
    if not hits:
        return Design(())
    # The linear screen can prefer a proxy pair; the logistic deviance of each candidate decides.
    def deviance(k: int) -> float:
        a, b = pairs[k]
        design = Design((feats.ids[a], feats.ids[b]), (feats.ids[a], feats.ids[b]), ((feats.ids[a], feats.ids[b]),))
        coef, intercept = _fit_logistic(design, feats)
        return log_loss(Hypothesis("", design, coef, intercept, 0.0).predict(feats), feats.y)
    hits = sorted(hits, key=deviance)[:2]
    products = tuple((feats.ids[pairs[k][0]], feats.ids[pairs[k][1]]) for k in hits)
    drivers = tuple(dict.fromkeys(f for pair in products for f in pair))
    return Design(drivers=drivers, columns=drivers, products=products)


def block(feats: Features) -> Design:
    """Correlated blocks (|r| >= 0.5) scored by their first principal component: a shared cause claim."""
    if not _usable(feats) or feats.x.shape[1] < 2:
        return Design(())
    corr = np.abs(np.corrcoef(feats.x, rowvar=False))
    np.fill_diagonal(corr, 0.0)
    unseen = set(range(feats.x.shape[1]))
    blocks: list[list[int]] = []
    while unseen:
        stack = [unseen.pop()]
        comp: list[int] = []
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in list(unseen):
                if corr[j, k] >= 0.5:
                    unseen.remove(k)
                    stack.append(k)
        if len(comp) >= 2:
            blocks.append(sorted(comp))
    if not blocks:
        return Design(())
    scores, loadings = [], []
    for comp in blocks:
        sub = feats.x[:, comp]
        _, _, vt = np.linalg.svd(sub - sub.mean(axis=0), full_matrices=False)
        load = vt[0]
        scores.append(sub @ load)
        loadings.append(load)
    p = _marginal_p(feats, np.column_stack(scores))
    best = int(np.argmin(p))
    if p[best] >= FDR / len(blocks):
        return Design(())
    comp, load = blocks[best], loadings[best]
    order = np.argsort(-np.abs(load))
    drivers = tuple(feats.ids[comp[j]] for j in order)[:MAX_DRIVERS]
    return Design(drivers=drivers, loadings={feats.ids[comp[j]]: float(load[j]) for j in range(len(comp))})


TEMPLATES: dict[str, Template] = {
    "null": null,
    "direct": direct,
    "conservative": conservative,
    "sparse": sparse,
    "confounder": confounder,
    "upstream": upstream,
    "interaction": interaction,
    "block": block,
}


# ---------------------------------------------------------------------- fitting


def _fit_logistic(design: Design, feats: Features) -> tuple[Array, float]:
    m = design.matrix(feats)
    if m.shape[1] == 0:
        rate = float(np.clip(feats.y.mean(), 1e-3, 1 - 1e-3))
        return np.empty(0), float(np.log(rate / (1 - rate)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = LogisticRegression(C=1.0, max_iter=500).fit(m, feats.y)
    return fit.coef_[0].astype(float), float(fit.intercept_[0])


def log_loss(p: Array, y: NDArray[np.int64]) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _subset(feats: Features, rows: NDArray[np.int64]) -> Features:
    return Features(feats.ids, feats.types, feats.x[rows], feats.y[rows], tuple(feats.stratum[i] for i in rows))


def fit_hypothesis(name: str, template: Template, feats: Features, seed: int = 0) -> Hypothesis:
    """Fit one template with cross-validated log loss that includes its selection step."""
    can_cv = feats.n >= 2 * CV_FOLDS and feats.y.sum() >= CV_FOLDS and (feats.n - feats.y.sum()) >= CV_FOLDS
    if can_cv:
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
        held = np.zeros(feats.n)
        for train, test in folds.split(feats.x, feats.y):
            sub = _subset(feats, train)
            design = template(sub)
            coef, intercept = _fit_logistic(design, sub)
            held[test] = Hypothesis(name, design, coef, intercept, 0.0).predict(_subset(feats, test))
        cv = log_loss(held, feats.y)
    else:
        cv = log_loss(np.full(feats.n, float(np.clip(feats.y.mean() if feats.n else 0.5, 1e-3, 1 - 1e-3))), feats.y)
    design = template(feats)
    coef, intercept = _fit_logistic(design, feats)
    return Hypothesis(name, design, coef, intercept, cv)


def fit_all(feats: Features, templates: dict[str, Template] | None = None, seed: int = 0) -> list[Hypothesis]:
    return [fit_hypothesis(name, t, feats, seed) for name, t in (templates or TEMPLATES).items()]
