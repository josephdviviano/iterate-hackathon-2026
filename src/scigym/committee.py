"""The committee over reaction networks: candidates, the probe by disagreement, refutation, the vote.

A probe is the experiment on which admitted members' simulated predictions
differ most. Members whose refitted model misses the observed result are
refuted. The committee's answer is the member closest to all others in
reaction-set distance (the plurality analogue for sets) and, separately, the
reactions a majority of members share.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from .data import Reaction, System
from .env import Experiment, Trajectory, simulate, smape
from .model import EPS, Hypothesis

FACTORS = (10.0, 0.1)


def candidates(system: System, baseline: Trajectory) -> list[Experiment]:
    """The experiments any arm may run: each species scaled by 10 and by 0.1 (a
    species that starts at 0 is set to its baseline peak and ten times it), and
    each non-zero species knocked out."""
    out = []
    for s in system.species:
        x0 = system.initial[s]
        if x0 > 0:
            out += [Experiment("set", s, x0 * f) for f in FACTORS]
        else:
            peak = float(np.max(baseline.values[s])) if s in baseline.values else 0.0
            v = peak if peak > 0 else 1.0
            out += [Experiment("set", s, v), Experiment("set", s, v * 10)]
    out += [Experiment("knockout", s) for s in system.species if system.initial[s] > 0]
    return out


def heldout(system: System) -> list[Experiment]:
    """Experiments no arm can run, for the trajectory error of the final model."""
    return [Experiment("observe")] + [Experiment("set", s, system.initial[s] * f)
                                      for s in system.species if system.initial[s] > 0 for f in (3.0, 0.3)]


def predictions(members: list[Hypothesis], system: System, exp: Experiment) -> list[Trajectory | None]:
    return [simulate(m.sbml, exp, system.t_end, system.n_steps) for m in members]


def disagreement(members: list[Hypothesis], system: System, exp: Experiment) -> float:
    """Mean pairwise SMAPE between members' predicted trajectories; a member that
    cannot be simulated is at distance 1 from every other."""
    preds = predictions(members, system, exp)
    if len(preds) < 2:
        return 0.0
    d = [1.0 if a is None or b is None else smape(a, b, system.species) for a, b in combinations(preds, 2)]
    return float(np.mean(d))


def choose_probe(members: list[Hypothesis], system: System, cands: list[Experiment], used: set) -> tuple[Experiment, float]:
    best, score = None, -1.0
    for c in cands:
        if c in used:
            continue
        d = disagreement(members, system, c)
        if d > score:
            best, score = c, d
    return best, score


def refute(members: list[Hypothesis], system: System, exp: Experiment, obs: Trajectory) -> list[bool]:
    """True where the member's prediction of the new experiment is within tolerance."""
    out = []
    for m in members:
        sim = simulate(m.sbml, exp, system.t_end, system.n_steps)
        out.append(sim is not None and smape(obs, sim, system.species) <= EPS)
    return out


def jaccard_distance(a: list, b: list) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return 1 - inter / union if union else 0.0


def vote(members: list[Hypothesis]) -> dict:
    """The medoid member, the majority reaction set, and the committee's disagreement."""
    if not members:
        return {"medoid": None, "majority": [], "spread": None, "n_distinct": 0}
    keys = [m.keys for m in members]
    n = len(members)
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist[i, j] = dist[j, i] = jaccard_distance(keys[i], keys[j])
    medoid = int(np.argmin(dist.sum(1)))
    counts = Counter(k for ks in keys for k in set(ks))
    majority = sorted(k for k, c in counts.items() if c / n > 0.5)
    spread = float(dist[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0
    return {"medoid": medoid, "majority": majority, "spread": spread,
            "n_distinct": len({tuple(sorted(ks)) for ks in keys})}


def rms(pred_keys: list, truth: tuple[Reaction, ...]) -> dict:
    """SciGym's reaction matching score: a reaction matches when its reactant and
    product multisets equal a reference reaction's; precision over the proposed
    reactions, recall over the reference ones."""
    cp, ct = Counter(pred_keys), Counter(r.key for r in truth)
    tp = sum((cp & ct).values())
    p = tp / sum(cp.values()) if cp else 0.0
    r = tp / sum(ct.values()) if ct else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def ste(sbml: str, system: System, exps: list[Experiment]) -> float:
    """Trajectory error of a model against the reference over held-out experiments."""
    errs = []
    for e in exps:
        truth = simulate(system.truth, e, system.t_end, system.n_steps)
        pred = simulate(sbml, e, system.t_end, system.n_steps) if sbml else None
        errs.append(1.0 if truth is None or pred is None else smape(truth, pred, system.species))
    return float(np.mean(errs))
