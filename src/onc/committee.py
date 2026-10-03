"""A committee of hypotheses over one ONC-AGI world.

Admission: a hypothesis is admitted when its cross-validated log loss is within
``tolerance`` of the best member. The null hypothesis is always admitted.
Weights: ``likelihood`` gives w ∝ exp(-beta · n · CV log loss); ``equal`` gives
every admitted member the same weight. Disagreement is the weighted entropy of
the distribution over distinct driver sets, normalised to [0, 1].
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from onc.hypotheses import TEMPLATES, Features, Hypothesis, Template, fit_all

CLUSTER_R = 0.8
TOLERANCE = 0.02
TAU = 0.5


@dataclass
class Committee:
    members: list[Hypothesis]
    weights: NDArray[np.float64]
    clusters: dict[str, int]
    dropped: list[str] = field(default_factory=list)

    @property
    def p_signal(self) -> float:
        return float(sum(w for h, w in zip(self.members, self.weights) if not h.is_null))

    def p_driver(self) -> dict[str, float]:
        """Weighted share of members that list each feature, one vote per cluster per member."""
        out: defaultdict[str, float] = defaultdict(float)
        for h, w in zip(self.members, self.weights):
            for f in dedupe_by_cluster(h.drivers, self.clusters):
                out[f] += float(w)
        return dict(out)

    def driver_sets(self) -> dict[frozenset[str], float]:
        sets: defaultdict[frozenset[str], float] = defaultdict(float)
        for h, w in zip(self.members, self.weights):
            sets[frozenset(dedupe_by_cluster(h.drivers, self.clusters))] += float(w)
        return dict(sets)

    @property
    def disagreement(self) -> float:
        """Normalised entropy over distinct driver sets; 0 when one set holds all weight."""
        if len(self.members) < 2:
            return 0.0
        p = np.array([v for v in self.driver_sets().values() if v > 0])
        return float(-(p * np.log(p)).sum() / np.log(len(self.members)))

    def predict(self, feats: Features) -> NDArray[np.float64]:
        return np.sum([w * h.predict(feats) for h, w in zip(self.members, self.weights)], axis=0)

    def member_predictions(self, feats: Features) -> NDArray[np.float64]:
        return np.vstack([h.predict(feats) for h in self.members])

    def ranking(self, tau: float = TAU) -> tuple[str, ...]:
        """Submission rule: abstain if P(signal) < 0.5, else features with P(driver) >= tau, decreasing."""
        if self.p_signal < 0.5:
            return ()
        p = self.p_driver()
        return tuple(sorted((f for f, v in p.items() if v >= tau), key=lambda f: (-p[f], self._mean_rank(f), f)))

    def _mean_rank(self, feature: str) -> float:
        """Weighted mean list position among the members that list the feature; breaks ties in P(driver)."""
        total, weight = 0.0, 0.0
        for h, w in zip(self.members, self.weights):
            if feature in h.drivers:
                total += w * h.drivers.index(feature)
                weight += w
        return total / weight if weight else float("inf")

    def summary(self) -> dict:
        return {
            "members": [
                {"name": h.name, "drivers": list(h.drivers), "cv_logloss": h.cv_logloss, "weight": float(w)}
                for h, w in zip(self.members, self.weights)
            ],
            "dropped": list(self.dropped),
            "p_signal": self.p_signal,
            "p_driver": self.p_driver(),
            "disagreement": self.disagreement,
        }


def dedupe_by_cluster(drivers: tuple[str, ...], clusters: dict[str, int]) -> list[str]:
    seen: set[int] = set()
    out: list[str] = []
    for f in drivers:
        c = clusters.get(f, hash(f))
        if c in seen:
            continue
        seen.add(c)
        out.append(f)
    return out


def cluster_features(feats: Features) -> dict[str, int]:
    """Complete-linkage clusters at |r| >= CLUSTER_R on the revealed data, as the scorer defines them."""
    p = feats.x.shape[1]
    if p < 2 or feats.n < 3:
        return {f: j for j, f in enumerate(feats.ids)}
    corr = np.abs(np.nan_to_num(np.corrcoef(feats.x, rowvar=False)))
    np.fill_diagonal(corr, 1.0)
    dist = squareform(np.clip(1.0 - corr, 0.0, None), checks=False)
    labels = fcluster(linkage(dist, method="complete"), t=1.0 - CLUSTER_R, criterion="distance")
    return {f: int(c) for f, c in zip(feats.ids, labels)}


def build_committee(
    feats: Features,
    *,
    weighting: str = "likelihood",
    beta: float = 1.0,
    tolerance: float = TOLERANCE,
    templates: dict[str, Template] | None = None,
    seed: int = 0,
) -> Committee:
    """Fit every template, admit by CV log loss, weight, and merge members with the same driver set."""
    fitted = fit_all(feats, templates or TEMPLATES, seed)
    best = min(h.cv_logloss for h in fitted)
    admitted, dropped = [], []
    for h in fitted:
        if h.is_null or h.cv_logloss <= best + tolerance:
            admitted.append(h)
        else:
            dropped.append(h.name)
    # One member per distinct driver set: the one with the lowest CV log loss speaks for it.
    by_set: dict[tuple[str, ...], Hypothesis] = {}
    for h in sorted(admitted, key=lambda h: h.cv_logloss):
        by_set.setdefault(tuple(sorted(h.drivers)), h)
    members = list(by_set.values())
    if weighting == "equal":
        weights = np.full(len(members), 1.0 / len(members))
    elif weighting == "likelihood":
        logw = np.array([-beta * feats.n * h.cv_logloss for h in members])
        weights = np.exp(logw - logw.max())
        weights /= weights.sum()
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return Committee(members, weights, cluster_features(feats), dropped)
