"""Uncertainty baselines from the dynamics-model literature, on the same splits.

All methods are compared at the effect level: for each held-out transition,
each object present before the action gets a predicted effect class (the
sorted list of fields that change, or no_change, gone). A transition is
correct when every object's class is right. Disagreement is the normalised
entropy over members that predict the same tuple of classes, as for the
program committee, whose predicted states are reduced to the same classes.

Baselines: nearest-neighbour copy of the most similar training object
transition (a floor with no uncertainty), a bagged committee of decision
trees (query by committee, Seung et al. 1992), and a deep ensemble of small
MLPs with different seeds (Lakshminarayanan et al. 2017). Members of the
ensembles are not filtered by exact train replay.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .committee import auroc, reliability
from .evaluate import load_runs
from .experiment import condition_dir
from .loader import CLICK, Transition, build_buffer, temporal_split
from .matrix import context_signature, effect_signature, object_type, pair_objects


@dataclass
class Vocab:
    types: list[str]
    actions: list[str]
    tags: list[str]
    classes: list[str]


def _tags(o: dict) -> list[str]:
    return [str(t) for t in (o.get("tags") or [])]


def build_vocab(train: list[Transition], all_transitions: list[Transition]) -> Vocab:
    types = sorted({object_type(o) for t in all_transitions for o in t.before_objs})
    actions = sorted({t.action_key for t in all_transitions})
    tags = [t for t, _ in Counter(tag for tr in all_transitions for o in tr.before_objs for tag in _tags(o)).most_common(24)]
    classes = sorted({sig for tr in train for sig in object_effects(tr).values()})
    return Vocab(types, actions, tags, classes)


def object_effects(t: Transition, after: list[dict] | None = None) -> dict[int, str]:
    """Effect class per before-object index. Born objects are not scored."""
    after = t.after_objs if after is None else after
    out: dict[int, str] = {}
    index = {id(o): i for i, o in enumerate(t.before_objs)}
    for bo, ao in pair_objects(t.before_objs, after):
        if bo is not None:
            out[index[id(bo)]] = effect_signature(bo, ao)
    return out


def features(t: Transition, i: int, vocab: Vocab) -> np.ndarray:
    o = t.before_objs[i]
    f: list[float] = []
    ty = object_type(o)
    f += [1.0 if ty == v else 0.0 for v in vocab.types]
    f += [1.0 if t.action_key == a else 0.0 for a in vocab.actions]
    f += [1.0 if tag in _tags(o) else 0.0 for tag in vocab.tags]
    x, y, w, h = (float(o.get(k, 0)) for k in ("x", "y", "w", "h"))
    f += [x / 64, y / 64, w / 64, h / 64, 1.0 if o.get("visible", True) else 0.0, float(o.get("layer", 0))]
    ctx = context_signature(t.before_objs, o, t.click)
    f += [1.0 if v in ctx.split(",") else 0.0 for v in vocab.types]
    f.append(1.0 if "hit" in ctx.split(",") else 0.0)
    if t.action_id == CLICK and t.click is not None:
        f += [(t.click[0] - x) / 64, (t.click[1] - y) / 64, 1.0]
    else:
        f += [0.0, 0.0, 0.0]
    return np.array(f, dtype=float)


def dataset(transitions: list[Transition], vocab: Vocab) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    X, y, where = [], [], []
    for ti, t in enumerate(transitions):
        eff = object_effects(t)
        for i in range(len(t.before_objs)):
            X.append(features(t, i, vocab))
            y.append(eff.get(i, "gone"))
            where.append((ti, i))
    return np.vstack(X), y, where


def _group_key(pred_by_obj: dict[int, str]) -> str:
    return json.dumps([pred_by_obj[i] for i in sorted(pred_by_obj)])


def transition_keys(preds_flat: list[str], where: list[tuple[int, int]], n_trans: int) -> list[str]:
    per: list[dict[int, str]] = [dict() for _ in range(n_trans)]
    for (ti, i), p in zip(where, preds_flat):
        per[ti][i] = p
    return [_group_key(d) for d in per]


def truth_keys(test: list[Transition]) -> list[str]:
    return [_group_key(object_effects(t)) for t in test]


def disagreement(keys_by_member: list[list[str]]) -> list[float]:
    k = len(keys_by_member)
    out = []
    for i in range(len(keys_by_member[0])):
        c = Counter(m[i] for m in keys_by_member)
        out.append(0.0 if len(c) < 2 else -sum(n / k * math.log(n / k) for n in c.values()) / math.log(k))
    return out


def plurality(keys_by_member: list[list[str]]) -> list[str]:
    return [Counter(m[i] for m in keys_by_member).most_common(1)[0][0] for i in range(len(keys_by_member[0]))]


def probes_to_falsify(keys_by_member: list[list[str]], truth: list[str], order: str, rng: random.Random | None = None
                      ) -> int | None:
    alive = list(range(len(keys_by_member)))
    left = list(range(len(truth)))
    rng = rng or random.Random(0)
    n = 0
    while left and alive:
        if order == "disagreement":
            d = disagreement([keys_by_member[m] for m in alive]) if len(alive) > 1 else [0.0] * len(truth)
            pick = max(left, key=lambda i: (d[i], -i))
        else:
            pick = rng.choice(left)
        left.remove(pick)
        n += 1
        alive = [m for m in alive if keys_by_member[m][pick] == truth[pick]]
    return n if not alive else None


def per_object_accuracy(pred_keys: list[str], truth: list[str]) -> float:
    hit = total = 0
    for p, t in zip(pred_keys, truth):
        pa, ta = json.loads(p) if p.startswith("[") else [], json.loads(t)
        total += len(ta)
        hit += sum(1 for a, b in zip(pa, ta) if a == b)
    return hit / total if total else 0.0


def survivors(keys_by_member: list[list[str]], truth: list[str], order: str, n: int = 5,
              rng: random.Random | None = None) -> list[int]:
    """Members still consistent after each of the first n probes."""
    alive = list(range(len(keys_by_member)))
    left = list(range(len(truth)))
    rng = rng or random.Random(0)
    out = [len(alive)]
    for _ in range(n):
        if not left or not alive:
            out.append(len(alive))
            continue
        if order == "disagreement" and len(alive) > 1:
            d = disagreement([keys_by_member[m] for m in alive])
            pick = max(left, key=lambda i: (d[i], -i))
        else:
            pick = rng.choice(left)
        left.remove(pick)
        alive = [m for m in alive if keys_by_member[m][pick] == truth[pick]]
        out.append(len(alive))
    return out


def score(name: str, keys_by_member: list[list[str]], truth: list[str], has_uncertainty: bool = True) -> dict:
    pred = plurality(keys_by_member)
    correct = [p == t for p, t in zip(pred, truth)]
    row = {"method": name, "members": len(keys_by_member), "accuracy": round(sum(correct) / len(truth), 4),
           "object_accuracy": round(per_object_accuracy(pred, truth), 4)}
    if has_uncertainty and len(keys_by_member) > 1:
        d = disagreement(keys_by_member)
        errors = [not c for c in correct]
        rel = {r["bin"]: r for r in reliability(d, errors)}
        split_n = len(truth) - rel["unanimous"]["n"]
        split_err = sum(e for e, x in zip(errors, d) if x > 0) / split_n if split_n else None
        rnd = [probes_to_falsify(keys_by_member, truth, "random", random.Random(s)) for s in range(20)]
        rnd_ok = [r for r in rnd if r is not None]
        row.update({
            "auroc": None if auroc(d, errors) is None else round(auroc(d, errors), 3),
            "unanimous_n": rel["unanimous"]["n"], "unanimous_error": rel["unanimous"]["error_rate"],
            "split_n": split_n, "split_error": None if split_err is None else round(split_err, 3),
            "probes_disagreement": probes_to_falsify(keys_by_member, truth, "disagreement"),
            "probes_random": round(sum(rnd_ok) / len(rnd_ok), 1) if rnd_ok else None,
            "survivors_disagreement": survivors(keys_by_member, truth, "disagreement"),
            "survivors_random_mean": [round(sum(x) / 20, 2) for x in zip(*[survivors(keys_by_member, truth, "random", rng=random.Random(s)) for s in range(20)])],
        })
    return row


def nearest_neighbour(Xtr: np.ndarray, ytr: list[str], Xte: np.ndarray) -> list[str]:
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return [ytr[j] for j in d.argmin(1)]


def tree_committee(Xtr, ytr, Xte, k: int, seed: int = 0) -> list[list[str]]:
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(seed)
    out = []
    for m in range(k):
        idx = rng.randint(0, len(ytr), len(ytr))
        clf = DecisionTreeClassifier(max_depth=6, random_state=seed + m).fit(Xtr[idx], [ytr[i] for i in idx])
        out.append(list(clf.predict(Xte)))
    return out


def mlp_ensemble(Xtr, ytr, Xte, k: int, seed: int = 0) -> list[list[str]]:
    from sklearn.neural_network import MLPClassifier

    out = []
    for m in range(k):
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=600, random_state=seed + m).fit(Xtr, ytr)
        out.append(list(clf.predict(Xte)))
    return out


def committee_keys(game: str, level: int, train_frac: float, condition: str, train, test,
                   test_level: int | None = None) -> list[list[str]]:
    cond = condition_dir(game, level, train_frac, condition, test_level)
    runs = load_runs(cond, train, test)
    keys = []
    for _name, m, _src, preds in runs:
        if not (m["consistent"] and preds):
            continue
        keys.append([_group_key(object_effects(t, p)) if p is not None else "<error>" for t, p in zip(test, preds)])
    return keys


def compare(game: str, level: int, train_frac: float, condition: str = "committee_devin", k: int = 8,
            test_level: int | None = None, seed: int = 0) -> list[dict]:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level)
    vocab = build_vocab(train, [t for t in transitions if t.level == level])
    Xtr, ytr, _ = dataset(train, vocab)
    Xte, _, where = dataset(test, vocab)
    truth = truth_keys(test)
    rows = []
    nn = transition_keys(nearest_neighbour(Xtr, ytr, Xte), where, len(test))
    rows.append(score("nearest neighbour (floor)", [nn], truth, has_uncertainty=False))
    trees = [transition_keys(p, where, len(test)) for p in tree_committee(Xtr, ytr, Xte, k, seed)]
    rows.append(score("bagged trees, query by committee", trees, truth))
    mlps = [transition_keys(p, where, len(test)) for p in mlp_ensemble(Xtr, ytr, Xte, k, seed)]
    rows.append(score("MLP deep ensemble", mlps, truth))
    ck = committee_keys(game, level, train_frac, condition, train, test, test_level)
    if ck:
        rows.append(score("program committee (effect level)", ck, truth))
    return rows


def print_rows(rows: list[dict]) -> None:
    cols = ["method", "members", "accuracy", "object_accuracy", "auroc", "unanimous_n", "unanimous_error", "split_n",
            "split_error", "probes_disagreement", "probes_random", "survivors_disagreement", "survivors_random_mean"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Effect-level comparison of uncertainty baselines with the committee.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--condition", default="committee_devin")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = compare(args.game, args.level, args.train_frac, args.condition, args.k, args.test_level, args.seed)
    out = condition_dir(args.game, args.level, args.train_frac, args.condition, args.test_level) / "baselines.json"
    out.write_text(json.dumps(rows, indent=1))
    print_rows(rows)


if __name__ == "__main__":
    main()
