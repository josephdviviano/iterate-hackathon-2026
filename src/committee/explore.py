"""Simulated exploration: which held-out transition to observe next.

The committee is a version space. Observing a transition removes every member
that mispredicted it. Query by committee (Seung, Opper and Sompolinsky, 1992)
observes the transition the members disagree on most. We compare it with a
random order and with OPINE-World's count-based priority, which observes the
transition whose effect rows have the fewest train counts. Members are filtered,
not resynthesized, so a probe that falsifies every member ends the run as a
"falsified" event: the committee has shown that it needs new hypotheses.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .committee import Committee, Member
from .loader import Transition
from .matrix import EffectMatrix, RowKey, context_signature, object_type
from .verify import canonical


@dataclass
class Trace:
    strategy: str
    probes: list[int]
    members_left: list[int]
    vote_error_left: list[float]
    falsified_at: int | None
    best_error_left: list[float] = field(default_factory=list)

    @property
    def probes_to_collapse(self) -> int | None:
        """Probes until exactly one member remains. None if the run ended by falsification or never collapsed."""
        for i, n in enumerate(self.members_left):
            if n == 1:
                return i
            if n == 0:
                return None
        return None


def _truth(test: list[Transition]) -> list[str]:
    return [json.dumps(canonical(t.after_objs)) for t in test]


def count_priority(train: list[Transition], test: list[Transition]) -> list[float]:
    """Higher for transitions whose object rows have fewer train observations."""
    matrix = EffectMatrix.from_transitions(train)
    scores = []
    for t in test:
        counts = []
        for bo in t.before_objs:
            key = RowKey(object_type(bo), t.action_key, context_signature(t.before_objs, bo, t.click))
            row = matrix.rows.get(key)
            counts.append(row.n if row else 0)
        scores.append(1.0 / (1.0 + min(counts)) if counts else 0.0)
    return scores


def simulate(members: list[Member], test: list[Transition], strategy: str, lam: float,
             train: list[Transition] | None = None, rng: random.Random | None = None) -> Trace:
    truth = _truth(test)
    alive = list(members)
    unobserved = list(range(len(test)))
    probes: list[int] = []
    members_left = [len(alive)]
    vote_error_left: list[float] = []
    falsified_at = None
    static_priority = count_priority(train, test) if strategy == "counts" and train else None
    rng = rng or random.Random(0)

    def vote_error(current: list[Member]) -> float:
        if not unobserved:
            return 0.0
        c = Committee(current, lam=lam)
        wrong = sum(c.vote(i).prediction != truth[i] for i in unobserved)
        return wrong / len(unobserved)

    def best_error(current: list[Member]) -> float:
        """Error of the survivor that is right most often on the unobserved transitions: selection in hindsight."""
        if not unobserved:
            return 0.0
        return min(sum(m.keys[i] != truth[i] for i in unobserved) for m in current) / len(unobserved)

    best_error_left = [best_error(alive)]
    vote_error_left.append(vote_error(alive))
    while unobserved and alive:
        if strategy == "disagreement":
            c = Committee(alive, lam=lam)
            pick = max(unobserved, key=lambda i: (c.vote(i).disagreement, -i))
        elif strategy == "counts":
            assert static_priority is not None
            pick = max(unobserved, key=lambda i: (static_priority[i], -i))
        else:
            pick = rng.choice(unobserved)
        probes.append(pick)
        unobserved.remove(pick)
        survivors = [m for m in alive if m.keys[pick] == truth[pick]]
        if not survivors:
            falsified_at = len(probes)
            members_left.append(0)
            vote_error_left.append(1.0)
            best_error_left.append(1.0)
            break
        alive = survivors
        members_left.append(len(alive))
        vote_error_left.append(vote_error(alive))
        best_error_left.append(best_error(alive))
    return Trace(strategy, probes, members_left, vote_error_left, falsified_at, best_error_left)


def compare_strategies(members: list[Member], train: list[Transition], test: list[Transition],
                       lam: float = 0.0, n_random: int = 20) -> dict:
    out: dict = {}
    for s in ("disagreement", "counts"):
        tr = simulate(members, test, s, lam, train)
        out[s] = {"probes_to_collapse": tr.probes_to_collapse, "falsified_at": tr.falsified_at,
                  "members_left": tr.members_left, "vote_error_left": [round(e, 3) for e in tr.vote_error_left]}
    randoms = [simulate(members, test, "random", lam, train, random.Random(k)) for k in range(n_random)]
    collapses = [t.probes_to_collapse for t in randoms if t.probes_to_collapse is not None]
    falsified = [t.falsified_at for t in randoms if t.falsified_at is not None]
    out["random"] = {
        "n_runs": n_random,
        "mean_probes_to_collapse": sum(collapses) / len(collapses) if collapses else None,
        "n_collapsed": len(collapses),
        "n_falsified": len(falsified),
        "mean_probes_to_falsify": sum(falsified) / len(falsified) if falsified else None,
    }
    return out
