"""Mixed-row disentanglement: which hidden condition splits a confounded effect row?

A row (type, action, context) with more than one observed effect means the
visible row key does not determine the outcome. Two ways to find the
missing condition, scored on the same footing:

A. OPINE-World's method, reimplemented: enumerate context features from a
   fixed vocabulary, split the row's training transitions by each feature's
   value, and accept the feature if it lowers the row's Dirichlet entropy
   by at least 0.05 with at least one identified sub-stratum.
B. Predicates read off the committee: perturb one feature of the before
   state and ask each admitted program again. A feature whose perturbation
   changes a program's predicted effect for the target object is a condition
   that program uses. Features are ranked by how often they flip predictions,
   then scored with the same entropy drop as A.

Both are checked out of sample: the entropy of the row's held-out effects
after conditioning on the chosen feature. Members that condition the same
row on different features show disagreement about the mechanism itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .evaluate import load_runs
from .experiment import condition_dir, require_objects
from .loader import CLICK, Transition, build_buffer, temporal_split
from .matrix import EffectMatrix, RowKey, context_signature, effect_signature, object_type, pair_objects
from .verify import run_program

SKIP_TARGET_FIELDS = {"name", "x", "y", "pixels", "type"}
ALPHA0 = 0.5
ACCEPT_DELTA = 0.05
N_MIN, MODAL_MIN = 2, 0.9


@dataclass(frozen=True)
class Feature:
    kind: str
    arg: tuple = ()

    def __str__(self) -> str:
        return self.kind + (f"({', '.join(map(str, self.arg))})" if self.arg else "")


def _bbox(o: dict) -> tuple[int, int, int, int]:
    x, y = int(o.get("x", 0)), int(o.get("y", 0))
    return x, y, x + int(o.get("w", 1)) - 1, y + int(o.get("h", 1)) - 1


def _hashable(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v


def _neighbours_at(state: list[dict], target: dict, dx: int, dy: int) -> list[dict]:
    """Objects touching the target on the side given by (dx, dy)."""
    x0, y0, x1, y1 = _bbox(target)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    px0, py0, px1, py1 = x0 + dx * w, y0 + dy * h, x1 + dx * w, y1 + dy * h
    out = []
    for o in state:
        if o is target or not o.get("visible", True):
            continue
        ox0, oy0, ox1, oy1 = _bbox(o)
        if ox0 <= px1 and ox1 >= px0 and oy0 <= py1 and oy1 >= py0:
            out.append(o)
    return out


def _within(state: list[dict], target: dict, r: int) -> list[dict]:
    x0, y0, x1, y1 = _bbox(target)
    out = []
    for o in state:
        if o is target or not o.get("visible", True):
            continue
        ox0, oy0, ox1, oy1 = _bbox(o)
        if ox0 <= x1 + r and ox1 >= x0 - r and oy0 <= y1 + r and oy1 >= y0 - r:
            out.append(o)
    return out


def feature_value(f: Feature, t: Transition, target: dict):
    if f.kind == "target_field":
        return _hashable(target.get(f.arg[0]))
    if f.kind == "pixels_hash":
        px = target.get("pixels")
        return hashlib.md5(json.dumps(px).encode()).hexdigest()[:6] if px is not None else None
    if f.kind == "neighbour_at_offset":
        return len(_neighbours_at(t.before_objs, target, *f.arg)) > 0
    if f.kind == "neighbourhood_radius":
        return len(_within(t.before_objs, target, f.arg[0]))
    if f.kind == "click_offset":
        if t.click is None:
            return None
        return (t.click[0] - int(target.get("x", 0)), t.click[1] - int(target.get("y", 0)))
    if f.kind == "global_field":
        ty, field = f.arg
        for o in t.before_objs:
            if object_type(o) == ty and o is not target:
                return _hashable(o.get(field))
        return None
    raise ValueError(f.kind)


def vocabulary(row_trans: list[tuple[Transition, dict]], action_id: int) -> list[Feature]:
    feats: list[Feature] = []
    fields: dict[str, set] = defaultdict(set)
    for t, bo in row_trans:
        for k, v in bo.items():
            if k not in SKIP_TARGET_FIELDS:
                fields[k].add(_hashable(v))
    feats += [Feature("target_field", (k,)) for k, vals in fields.items() if len(vals) >= 2]
    if any(bo.get("pixels") is not None for _, bo in row_trans):
        feats.append(Feature("pixels_hash"))
    feats += [Feature("neighbour_at_offset", (dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    feats += [Feature("neighbourhood_radius", (2,)), Feature("neighbourhood_radius", (3,))]
    if action_id == CLICK:
        feats.append(Feature("click_offset"))
    # fields of other object types, for HUD and counter conditions
    others: dict[tuple[str, str], set] = defaultdict(set)
    for t, bo in row_trans:
        for o in t.before_objs:
            if o is bo:
                continue
            for k, v in o.items():
                if k not in {"name", "pixels"}:
                    others[(object_type(o), k)].add(_hashable(v))
    feats += [Feature("global_field", key) for key, vals in others.items() if len(vals) >= 2][:12]
    return feats


def row_transitions(train: list[Transition], key: RowKey) -> list[tuple[Transition, dict, dict | None]]:
    out = []
    for t in train:
        if t.action_key != key.action:
            continue
        for bo, ao in pair_objects(t.before_objs, t.after_objs):
            if bo is None:
                continue
            k = RowKey(object_type(bo), t.action_key, context_signature(t.before_objs, bo, t.click))
            if k == key:
                out.append((t, bo, ao))
    return out


def entropy(counter: Counter, alphabet_size: int) -> float:
    n = sum(counter.values())
    E = max(2, alphabet_size)
    denom = E * ALPHA0 + n
    h = 0.0
    for c in counter.values():
        p = (ALPHA0 + c) / denom
        h -= p * math.log(p)
    unseen = E - len(counter)
    if unseen > 0:
        p0 = ALPHA0 / denom
        h -= unseen * p0 * math.log(p0)
    return h / math.log(E)


def split_score(trans: list[tuple[Transition, dict, dict | None]], f: Feature, alphabet_size: int) -> dict:
    strata: dict = defaultdict(Counter)
    for t, bo, ao in trans:
        strata[feature_value(f, t, bo)][effect_signature(bo, ao)] += 1
    total = sum(sum(c.values()) for c in strata.values())
    eta_new = sum(sum(c.values()) * entropy(c, alphabet_size) for c in strata.values()) / total
    identified = sum(1 for c in strata.values() if sum(c.values()) >= N_MIN and c.most_common(1)[0][1] / sum(c.values()) >= MODAL_MIN)
    return {"eta_new": round(eta_new, 4), "n_substrata": len(strata), "identified": identified}


def perturbations(f: Feature, t: Transition, bo: dict, value_pool: dict) -> list[tuple[list[dict], int | dict]]:
    """Before states with one feature of the target changed. Returns (state, action) pairs."""
    state = copy.deepcopy(t.before_objs)
    idx = t.before_objs.index(bo)
    target = state[idx]
    outs = []
    if f.kind == "target_field":
        field = f.arg[0]
        for v in value_pool.get(("target", field), []):
            if _hashable(v) != _hashable(target.get(field)):
                s2 = copy.deepcopy(state)
                s2[idx][field] = v
                outs.append((s2, t.action))
                if len(outs) >= 2:
                    break
    elif f.kind == "pixels_hash" and target.get("pixels"):
        s2 = copy.deepcopy(state)
        px = s2[idx]["pixels"]
        if isinstance(px, list) and px and isinstance(px[0], list) and px[0]:
            px[0][0] = (int(px[0][0]) + 1) % 16
            outs.append((s2, t.action))
    elif f.kind == "neighbour_at_offset":
        near = _neighbours_at(state, target, *f.arg)
        if near:
            s2 = [o for o in state if not any(o is n for n in near)]
            outs.append((s2, t.action))
    elif f.kind == "neighbourhood_radius":
        near = _within(state, target, f.arg[0])
        if near:
            s2 = [o for o in state if not any(o is n for n in near)]
            outs.append((s2, t.action))
    elif f.kind == "click_offset" and t.click is not None:
        x, y = t.click
        for dx, dy in ((1, 0), (0, 1), (-1, 0)):
            outs.append((copy.deepcopy(state), {"action_id": CLICK, "x": x + dx, "y": y + dy}))
    elif f.kind == "global_field":
        ty, field = f.arg
        for j, o in enumerate(state):
            if j != idx and object_type(o) == ty:
                for v in value_pool.get(("global", ty, field), []):
                    if _hashable(v) != _hashable(o.get(field)):
                        s2 = copy.deepcopy(state)
                        s2[j][field] = v
                        outs.append((s2, t.action))
                        break
                break
    return outs


def value_pools(train: list[Transition]) -> dict:
    pool: dict = defaultdict(set)
    for t in train:
        for o in t.before_objs:
            for k, v in o.items():
                if k not in {"name", "pixels"}:
                    pool[("target", k)].add(_hashable(v))
                    pool[("global", object_type(o), k)].add(_hashable(v))
    return {k: [json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v for v in vals] for k, vals in pool.items()}


def predicted_effect(pred: list[dict] | None, before: list[dict], target_index: int) -> str:
    if pred is None:
        return "<error>"
    target = before[target_index]
    for bo, ao in pair_objects(before, pred):
        if bo is target:
            return effect_signature(bo, ao)
    return "gone"


def analyse(game: str, level: int, train_frac: float, condition: str, test_level: int | None = None,
            max_rows: int = 12) -> dict:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level)
    cond = condition_dir(game, level, train_frac, condition, test_level)
    require_objects(cond, "perturbation attribution")
    members = [(name, src) for name, m, src, _ in load_runs(cond, train, test) if m["consistent"]]
    matrix = EffectMatrix.from_transitions(train)
    alphabet = len(matrix.alphabet)
    pools = value_pools(train)
    results = []
    for row in matrix.mixed_rows()[:max_rows]:
        key = row.key
        trans = row_transitions(train, key)
        if len(trans) < 2:
            continue
        eta_old = round(matrix.eta(key) or 0.0, 4)
        feats = vocabulary([(t, bo) for t, bo, _ in trans], int(key.action.replace("ACTION", "")) if key.action != "RESET" else 0)
        # A: enumerate and score
        a_scores = []
        for f in feats:
            sc = split_score(trans, f, alphabet)
            sc.update({"feature": str(f), "delta": round(eta_old - sc["eta_new"], 4),
                       "accepted": (eta_old - sc["eta_new"]) >= ACCEPT_DELTA and sc["identified"] >= 1})
            a_scores.append(sc)
        a_scores.sort(key=lambda d: -d["delta"])
        # B: perturb each feature, ask each member
        jobs: list[tuple[int, Feature, Transition, dict, list[dict], int | dict]] = []
        base_rows: list[Transition] = []
        for ti, (t, bo, _) in enumerate(trans):
            base_rows.append(Transition(t.step, t.level, t.action_id, t.click, False, [], [], t.before_objs, []))
            for f in feats:
                for state, action in perturbations(f, t, bo, pools):
                    aid = action["action_id"] if isinstance(action, dict) else action
                    click = (action["x"], action["y"]) if isinstance(action, dict) else None
                    jobs.append((ti, f, t, bo, state, action))
        flips: dict[str, Counter] = defaultdict(Counter)  # feature -> member -> flips
        tries: dict[str, int] = Counter()
        member_features: dict[str, set] = defaultdict(set)
        for name, src in members:
            probe_rows = base_rows + [
                Transition(t.step, t.level, (a["action_id"] if isinstance(a, dict) else a),
                           ((a["x"], a["y"]) if isinstance(a, dict) else None), False, [], [], state, [])
                for (_, _, t, _, state, a) in jobs]
            preds = run_program(src, [], probe_rows, timeout_s=300).test_preds
            base_eff = [predicted_effect(preds[i], trans[i][0].before_objs, trans[i][0].before_objs.index(trans[i][1]))
                        for i in range(len(trans))]
            for j, (ti, f, t, bo, state, a) in enumerate(jobs):
                p = preds[len(trans) + j]
                tgt_idx = next((k for k, o in enumerate(state) if o.get("name") == bo.get("name") and o.get("x") == bo.get("x") and o.get("y") == bo.get("y")), None)
                if tgt_idx is None:
                    continue
                eff = predicted_effect(p, state, tgt_idx)
                tries[str(f)] += 1
                if eff != base_eff[ti] and eff != "<error>":
                    flips[str(f)][name] += 1
                    member_features[name].add(str(f))
        b_scores = []
        for f in feats:
            n_try = tries.get(str(f), 0)
            n_flip = sum(flips[str(f)].values())
            if n_try == 0:
                continue
            sc = split_score(trans, f, alphabet)
            sc.update({"feature": str(f), "sensitivity": round(n_flip / n_try, 3), "members_using": len(flips[str(f)]),
                       "delta": round(eta_old - sc["eta_new"], 4),
                       "accepted": (eta_old - sc["eta_new"]) >= ACCEPT_DELTA and sc["identified"] >= 1})
            b_scores.append(sc)
        b_scores.sort(key=lambda d: (-d["sensitivity"], -d["delta"]))
        # out of sample: held-out transitions of this row, entropy before and after conditioning
        test_trans = row_transitions(test, key)
        def oos(feature_name: str | None) -> float | None:
            if not test_trans:
                return None
            if feature_name is None:
                return round(entropy(Counter(effect_signature(bo, ao) for _, bo, ao in test_trans), alphabet), 4)
            f = next(ff for ff in feats if str(ff) == feature_name)
            return split_score(test_trans, f, alphabet)["eta_new"]
        # mechanism disagreement: do members condition this row on the same features?
        sets = [frozenset(member_features[name]) for name, _ in members]
        distinct_sets = len(set(sets))
        results.append({
            "row": str(key), "n_train": row.n, "effects": dict(row.effects), "eta_old": eta_old,
            "n_test_objects": len(test_trans), "oos_eta_unconditioned": oos(None),
            "A_best": a_scores[0] if a_scores else None,
            "A_oos_eta": oos(a_scores[0]["feature"]) if a_scores else None,
            "B_best": b_scores[0] if b_scores else None,
            "B_oos_eta": oos(b_scores[0]["feature"]) if b_scores else None,
            "B_top3": b_scores[:3],
            "members_feature_sets": {name: sorted(member_features[name]) for name, _ in members},
            "mechanism_distinct_feature_sets": distinct_sets,
        })
    summary = {
        "game": game, "level": level, "n_mixed_rows": len(results), "n_members": len(members),
        "A_rows_accepted": sum(1 for r in results if r["A_best"] and r["A_best"]["accepted"]),
        "B_rows_accepted": sum(1 for r in results if r["B_best"] and r["B_best"]["accepted"]),
        "A_mean_oos_eta": _mean([r["A_oos_eta"] for r in results]),
        "B_mean_oos_eta": _mean([r["B_oos_eta"] for r in results]),
        "unconditioned_mean_oos_eta": _mean([r["oos_eta_unconditioned"] for r in results]),
        "rows_where_A_and_B_pick_same_feature": sum(1 for r in results if r["A_best"] and r["B_best"] and r["A_best"]["feature"] == r["B_best"]["feature"]),
        "rows": results,
    }
    (cond / "disentangle.json").write_text(json.dumps(summary, indent=1))
    return summary


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def print_summary(s: dict) -> None:
    print(f"{s['game']} L{s['level']}: {s['n_mixed_rows']} mixed rows, {s['n_members']} members")
    print(f"  rows resolved (delta eta >= 0.05, >= 1 identified sub-stratum): A {s['A_rows_accepted']}, B {s['B_rows_accepted']}; "
          f"same feature chosen on {s['rows_where_A_and_B_pick_same_feature']} rows")
    print(f"  held-out row entropy: unconditioned {s['unconditioned_mean_oos_eta']}, after A {s['A_mean_oos_eta']}, after B {s['B_mean_oos_eta']}")
    for r in s["rows"]:
        a = r["A_best"]; b = r["B_best"]
        print(f"  - {r['row']}  n={r['n_train']} eta={r['eta_old']} effects={r['effects']}")
        print(f"      A: {a['feature'] if a else None} delta={a['delta'] if a else None} acc={a['accepted'] if a else None} oos={r['A_oos_eta']}")
        print(f"      B: {b['feature'] if b else None} sens={b['sensitivity'] if b else None} members={b['members_using'] if b else None} delta={b['delta'] if b else None} oos={r['B_oos_eta']}  distinct member feature sets={r['mechanism_distinct_feature_sets']}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Split mixed effect rows: OPINE-World's enumerator vs committee predicates.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--condition", default="committee_devin")
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=12)
    args = parser.parse_args(argv)
    print_summary(analyse(args.game, args.level, args.train_frac, args.condition, args.test_level, args.max_rows))


if __name__ == "__main__":
    main()
