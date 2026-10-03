"""A weighted committee of exact-replay-consistent programs.

Every member replays the train transitions exactly, so likelihood cannot
separate them. By default all members weigh the same: the committee predicts
by plurality over the members' next states and reports the normalised
entropy of that vote as its uncertainty. An optional simplicity prior,
w_i proportional to exp(-lambda * L_i) with L_i the gzip length of the
stripped source (Rissanen, 1978), is kept for ablation; on this data it did
not track held-out accuracy (RESULTS R14).
"""

from __future__ import annotations

import ast
import gzip
import json
import math
from dataclasses import dataclass
from functools import cached_property

from .verify import canonical


def normalized_source(source: str) -> str:
    """Source with comments and docstrings removed, for a length that measures mechanics, not prose."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def description_length(source: str) -> int:
    try:
        text = normalized_source(source)
    except SyntaxError:
        text = source
    return len(gzip.compress(text.encode(), compresslevel=9))


@dataclass
class Member:
    name: str
    source: str
    test_preds: list[list[dict] | None]
    length: int

    @cached_property
    def keys(self) -> list[str]:
        """Canonical JSON of each prediction, computed once per member."""
        return [json.dumps(canonical(p)) if p is not None else "<error>" for p in self.test_preds]


@dataclass
class Vote:
    distribution: dict[str, float]
    prediction: str
    disagreement: float

    @property
    def n_distinct(self) -> int:
        return len(self.distribution)


class Committee:
    def __init__(self, members: list[Member], lam: float = 0.0):
        if not members:
            raise ValueError("a committee needs at least one member")
        self.members = members
        self.lam = lam
        lengths = [m.length for m in members]
        base = min(lengths)
        raw = [math.exp(-lam * (l - base)) for l in lengths]
        z = sum(raw)
        self.weights = [r / z for r in raw]

    def vote(self, index: int) -> Vote:
        dist: dict[str, float] = {}
        for m, w in zip(self.members, self.weights):
            key = m.keys[index]
            dist[key] = dist.get(key, 0.0) + w
        prediction = max(dist, key=lambda k: dist[k])
        if len(dist) > 1:
            h = -sum(p * math.log(p) for p in dist.values() if p > 0)
            disagreement = h / math.log(len(self.members))
        else:
            disagreement = 0.0
        return Vote(dist, prediction, disagreement)

    def votes(self) -> list[Vote]:
        n = len(self.members[0].test_preds)
        return [self.vote(i) for i in range(n)]

    def uniform_disagreement(self, index: int) -> float:
        """Normalised entropy of the members' predictions with equal weights. The MDL weights
        sharpen the point prediction; equal weights keep every surviving hypothesis visible."""
        counts: dict[str, int] = {}
        for m in self.members:
            key = m.keys[index]
            counts[key] = counts.get(key, 0) + 1
        if len(counts) < 2:
            return 0.0
        n = len(self.members)
        return -sum(c / n * math.log(c / n) for c in counts.values()) / math.log(n)

    def evaluate(self, test_after: list[list[dict]]) -> dict:
        votes = self.votes()
        truth = [json.dumps(canonical(a)) for a in test_after]
        correct = [v.prediction == t for v, t in zip(votes, truth)]
        simplest = self.members[self.weights.index(max(self.weights))]
        simplest_correct = [k == t for k, t in zip(simplest.keys, truth)]
        member_acc = [sum(k == t for k, t in zip(m.keys, truth)) / max(1, len(truth)) for m in self.members]
        disagreement = [v.disagreement for v in votes]
        uniform = [self.uniform_disagreement(i) for i in range(len(votes))]
        errors = [not c for c in correct]
        return {
            "n_members": len(self.members),
            "weights": [round(w, 4) for w in self.weights],
            "lengths": [m.length for m in self.members],
            "vote_accuracy": sum(correct) / max(1, len(correct)),
            "simplest_accuracy": sum(simplest_correct) / max(1, len(simplest_correct)),
            "mean_member_accuracy": sum(member_acc) / len(member_acc),
            "member_accuracy": [round(a, 4) for a in member_acc],
            "auroc_disagreement_vs_error": auroc(disagreement, errors),
            "auroc_uniform_disagreement_vs_error": auroc(uniform, errors),
            "reliability": reliability(disagreement, errors),
            "reliability_uniform": reliability(uniform, errors),
            "mean_disagreement": sum(disagreement) / max(1, len(disagreement)),
            "n_distinct_behaviours": len({tuple(m.keys) for m in self.members}),
            "per_transition": [{"disagreement": round(d, 4), "uniform_disagreement": round(u, 4), "correct": c,
                                "n_distinct": v.n_distinct}
                               for d, u, c, v in zip(disagreement, uniform, correct, votes)],
        }


def reliability(disagreement: list[float], errors: list[bool]) -> list[dict]:
    """Error rate per disagreement bin: unanimous, low, medium, high."""
    edges = [(0.0, 0.0, "unanimous"), (0.0, 1 / 3, "low"), (1 / 3, 2 / 3, "medium"), (2 / 3, 1.0001, "high")]
    out = []
    for lo, hi, name in edges:
        idx = [i for i, d in enumerate(disagreement) if (d == 0.0 if hi == 0.0 else lo < d <= hi)]
        out.append({"bin": name, "n": len(idx),
                    "error_rate": round(sum(errors[i] for i in idx) / len(idx), 4) if idx else None})
    return out


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Probability that a random positive scores above a random negative, ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def row_uncertainty(committee: Committee, train, test) -> list[dict]:
    """Per (type, action, context) row touched by the test transitions: the count-based
    ontology error from train, and the committee's weighted effect entropy. Rows unseen in
    train have no count-based value; the committee still gives one."""
    from .matrix import EffectMatrix, RowKey, context_signature, effect_signature, object_type, pair_objects

    matrix = EffectMatrix.from_transitions(train)
    dists: dict[RowKey, dict[str, float]] = {}
    actual: dict[RowKey, dict[str, int]] = {}
    for i, t in enumerate(test):
        preds = [(m, w) for m, w in zip(committee.members, committee.weights)]
        pred_pairs = {id(m): pair_objects(t.before_objs, m.test_preds[i]) if m.test_preds[i] is not None else None
                      for m, _ in preds}
        true_pairs = pair_objects(t.before_objs, t.after_objs)
        for bo, ao in true_pairs:
            if bo is None:
                continue
            key = RowKey(object_type(bo), t.action_key, context_signature(t.before_objs, bo, t.click))
            actual.setdefault(key, {})
            sig = effect_signature(bo, ao)
            actual[key][sig] = actual[key].get(sig, 0) + 1
            d = dists.setdefault(key, {})
            for m, w in preds:
                pp = pred_pairs[id(m)]
                if pp is None:
                    psig = "<error>"
                else:
                    psig = next((effect_signature(b, a) for b, a in pp if b is bo), "gone")
                d[psig] = d.get(psig, 0.0) + w
    rows = []
    for key, d in dists.items():
        z = sum(d.values())
        q = {k: v / z for k, v in d.items()}
        h = -sum(p * math.log(p) for p in q.values() if p > 0)
        eta_committee = h / math.log(len(q)) if len(q) > 1 else 0.0  # normalised over the row's own support
        row = matrix.rows.get(key)
        rows.append({
            "row": str(key),
            "n_train": row.n if row else 0,
            "eta_counts": matrix.eta(key),
            "eta_committee": round(eta_committee, 4),
            "committee_effects": {k: round(v, 3) for k, v in sorted(q.items(), key=lambda kv: -kv[1])},
            "observed_effects": actual[key],
        })
    rows.sort(key=lambda r: (-r["eta_committee"], r["n_train"]))
    return rows


def row_calibration(committee: Committee, train, test) -> dict:
    """Mechanism-level check: does a row's committee entropy predict the committee's error on
    that row? Each row touched by the held-out transitions gets eta_committee (from
    row_uncertainty) and the share of its objects whose plurality effect is wrong."""
    from .matrix import RowKey, context_signature, effect_signature, object_type, pair_objects

    rows = {r["row"]: r for r in row_uncertainty(committee, train, test)}
    wrong: dict[str, int] = {}
    total: dict[str, int] = {}
    for i, t in enumerate(test):
        true_pairs = {id(bo): ao for bo, ao in pair_objects(t.before_objs, t.after_objs) if bo is not None}
        votes: dict[int, dict[str, float]] = {}
        for m, w in zip(committee.members, committee.weights):
            pred = m.test_preds[i]
            if pred is None:
                continue
            for bo, ao in pair_objects(t.before_objs, pred):
                if bo is not None:
                    d = votes.setdefault(id(bo), {})
                    sig = effect_signature(bo, ao)
                    d[sig] = d.get(sig, 0.0) + w
        for bo in t.before_objs:
            key = str(RowKey(object_type(bo), t.action_key, context_signature(t.before_objs, bo, t.click)))
            if key not in rows:
                continue
            truth = effect_signature(bo, true_pairs.get(id(bo)))
            v = votes.get(id(bo))
            plural = max(v, key=lambda k: v[k]) if v else None
            total[key] = total.get(key, 0) + 1
            wrong[key] = wrong.get(key, 0) + (plural != truth)
    table = []
    for key, r in rows.items():
        if total.get(key, 0) == 0:
            continue
        table.append({"row": key, "n_train": r["n_train"], "eta_counts": r["eta_counts"], "eta_committee": r["eta_committee"],
                      "n_test_objects": total[key], "error_rate": round(wrong[key] / total[key], 4)})
    etas = [t["eta_committee"] for t in table]
    errs = [t["error_rate"] > 0 for t in table]
    bins = []
    for lo, hi, name in ((0.0, 0.0, "zero"), (0.0, 0.5, "low"), (0.5, 1.0001, "high")):
        sel = [t for t in table if (t["eta_committee"] == 0.0 if hi == 0.0 else lo < t["eta_committee"] <= hi)]
        bins.append({"bin": name, "n_rows": len(sel),
                     "mean_error_rate": round(sum(t["error_rate"] for t in sel) / len(sel), 4) if sel else None,
                     "rows_with_error": sum(t["error_rate"] > 0 for t in sel)})
    unseen = [t for t in table if t["n_train"] == 0]
    seen = [t for t in table if t["n_train"] > 0]
    return {
        "n_rows": len(table), "n_unseen_rows": len(unseen),
        "auroc_eta_committee_vs_row_error": auroc(etas, errs),
        "auroc_unseen_rows_only": auroc([t["eta_committee"] for t in unseen], [t["error_rate"] > 0 for t in unseen]),
        "auroc_eta_counts_seen_rows": auroc([t["eta_counts"] for t in seen], [t["error_rate"] > 0 for t in seen]) if seen else None,
        "auroc_eta_committee_seen_rows": auroc([t["eta_committee"] for t in seen], [t["error_rate"] > 0 for t in seen]) if seen else None,
        "bins": bins, "rows": sorted(table, key=lambda t: -t["eta_committee"]),
    }
