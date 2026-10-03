"""Targeted synthesis: new members are asked to settle what the committee disputes.

The probe set is every training state paired with every action the agent
could take there, including a click on each visible object. The current
members predict all probes. The probes with the most disagreement, with the
competing predicted outcomes written out, become the next member's seed.
Admission is unchanged, so a new member may land on an existing behaviour;
that is convergence, not a failure. Held-out transitions never enter the
probe set.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter

from .evaluate import load_runs
from .experiment import condition_dir, describe, pack_run, write_run
from .loader import CLICK, RESET, Transition, build_buffer, temporal_split
from .matrix import effect_signature, pair_objects
from .synth_api import synthesize_any
from .verify import canonical, run_program


def action_set(transitions: list[Transition]) -> list[int]:
    return sorted({t.action_id for t in transitions if t.action_id != RESET})


def probe_set(train: list[Transition], actions: list[int], max_probes: int = 400) -> list[Transition]:
    """Hypothetical probes: each distinct training state with each action. Clicks target object centres."""
    seen: set[str] = set()
    probes: list[Transition] = []
    for t in train:
        key = json.dumps(canonical(t.before_objs))
        if key in seen:
            continue
        seen.add(key)
        for a in actions:
            if a == CLICK:
                for o in t.before_objs:
                    if o.get("visible", True):
                        cx = int(o.get("x", 0)) + int(o.get("w", 1)) // 2
                        cy = int(o.get("y", 0)) + int(o.get("h", 1)) // 2
                        probes.append(Transition(t.step, t.level, CLICK, (cx, cy), False, [], [], t.before_objs, []))
            else:
                probes.append(Transition(t.step, t.level, a, None, False, [], [], t.before_objs, []))
    if len(probes) > max_probes:
        stride = len(probes) / max_probes
        probes = [probes[int(i * stride)] for i in range(max_probes)]
    return probes


def predict_all(sources: list[str], probes: list[Transition]) -> list[list[list[dict] | None]]:
    return [run_program(src, [], probes).test_preds for src in sources]


def _outcome_text(before: list[dict], pred: list[dict] | None, limit: int = 3) -> str:
    if pred is None:
        return "error"
    parts = []
    for bo, ao in pair_objects(before, pred):
        sig = effect_signature(bo, ao)
        if sig == "no_change":
            continue
        name = (bo or ao).get("name")
        if bo is None:
            parts.append(f"{name} appears")
        elif ao is None:
            parts.append(f"{name} disappears")
        else:
            changes = ", ".join(f"{k} {bo.get(k)}->{ao.get(k)}" for k in sig.split(",") if k != "pixels")
            parts.append(f"{name}: {changes or 'pixels change'}")
        if len(parts) >= limit:
            break
    return "; ".join(parts) if parts else "nothing changes"


def disputed_probes(probes: list[Transition], preds: list[list[list[dict] | None]], top: int = 6) -> list[dict]:
    """Probes ranked by equal-weight disagreement, with the competing outcomes and their support."""
    k = len(preds)
    out = []
    for i, p in enumerate(probes):
        groups: Counter = Counter()
        sample: dict[str, list[dict] | None] = {}
        for member in preds:
            pred = member[i]
            key = json.dumps(canonical(pred)) if pred is not None else "<error>"
            groups[key] += 1
            sample.setdefault(key, pred)
        if len(groups) < 2:
            continue
        h = -sum(c / k * math.log(c / k) for c in groups.values()) / math.log(k)
        out.append({
            "step": p.step, "action": p.action, "disagreement": round(h, 3),
            "outcomes": [(c, _outcome_text(p.before_objs, sample[key])) for key, c in groups.most_common()],
        })
    out.sort(key=lambda d: -d["disagreement"])
    return out[:top]


def member_headers(sources: list[str]) -> list[str]:
    heads = []
    for src in sources:
        text = src.strip()
        if text.startswith('"""') or text.startswith("'''"):
            q = text[:3]
            end = text.find(q, 3)
            heads.append(text[3:end].strip()[:400] if end > 0 else "")
        else:
            lines = [l.lstrip("# ").rstrip() for l in text.splitlines() if l.startswith("#")]
            heads.append(" ".join(lines[:6])[:400])
    return heads


def targeted_seed(disputes: list[dict], headers: list[str], n_members: int) -> str:
    lines = [f"{n_members} programs already replay every observed transition exactly. They disagree about "
             f"what would happen in these situations, which no observation settles. Each line gives the "
             f"situation (training step whose before state is used, and the action), then the predicted "
             f"outcomes with how many programs predict each:"]
    for d in disputes:
        lines.append(f"- step {d['step']}, action {json.dumps(d['action'])}: " +
                     " | ".join(f"{c} predict: {txt}" for c, txt in d["outcomes"]))
    lines.append("")
    lines.append("Their stated mechanics, one per program:")
    for i, h in enumerate(headers):
        if h:
            lines.append(f"- program {i}: {h}")
    lines.append("")
    lines.append("Write the program whose mechanics decide these situations in the way the observed "
                 "transitions make most plausible. Prefer the simplest rule set that explains all observations "
                 "and takes a definite stance on each disputed situation. If one of the existing stances is "
                 "clearly right, adopt it; you do not have to differ for its own sake.")
    return "\n".join(lines)


def run_active(game: str, level: int, train_frac: float, condition: str, init_from: str, init_runs: int,
               total: int, per_round: int, cfg: dict, test_level: int | None = None,
               max_probes: int = 400, top: int = 6) -> None:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level)
    probes = probe_set(train, action_set([t for t in transitions if t.level == level]), max_probes)
    base = condition_dir(game, level, train_frac, condition, test_level)
    src_dir = condition_dir(game, level, train_frac, init_from, test_level)
    base.mkdir(parents=True, exist_ok=True)
    for k in range(init_runs):
        if not (base / f"run{k}").exists():
            shutil.copytree(src_dir / f"run{k}", base / f"run{k}")
    print(f"{game} L{level} {condition}: {len(train)} train, {len(test)} test, {len(probes)} probes, "
          f"{init_runs} members copied from {init_from}", flush=True)
    rounds = []
    while True:
        runs = load_runs(base, train, test)
        members = [(name, src) for name, m, src, _ in runs if m["consistent"]]
        sources = [src for _, src in members]
        preds = predict_all(sources, probes)
        disputes = disputed_probes(probes, preds, top=top)
        mean_dis = (sum(d["disagreement"] for d in disputed_probes(probes, preds, top=len(probes))) / len(probes))
        rounds.append({"n_runs": len(runs), "n_members": len(members), "mean_probe_disagreement": round(mean_dis, 4),
                       "n_disputed": sum(1 for d in disputed_probes(probes, preds, top=len(probes)) if d["disagreement"] > 0)})
        (base / "active_rounds.json").write_text(json.dumps(rounds, indent=1))
        print(f"  members={len(members)} probes disputed={rounds[-1]['n_disputed']}/{len(probes)} "
              f"mean disagreement={mean_dis:.3f}", flush=True)
        if len(runs) >= total:
            break
        if not disputes:
            print("  no disputed probes left; stopping early", flush=True)
            break
        seed = targeted_seed(disputes, member_headers(sources), len(members))
        batch = min(per_round, total - len(runs))
        from concurrent.futures import ThreadPoolExecutor

        def job(k: int) -> None:
            result = synthesize_any(train, seed, cfg)
            verdict = run_program(result.source, train, test)
            record = pack_run(result, verdict)
            record["meta"]["round"] = len(rounds)
            record["meta"]["disputes"] = disputes
            meta = write_run(base / f"run{k}", record)
            print(describe(meta, f"{game} L{level} {condition} run{k}"), flush=True)

        with ThreadPoolExecutor(max_workers=batch) as pool:
            list(pool.map(job, range(len(runs), len(runs) + batch)))


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .experiment import add_backend_args, backend_cfg

    parser = argparse.ArgumentParser(description="Grow a committee by synthesizing where it disagrees.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--condition", default="active_devin")
    parser.add_argument("--init-from", default="committee_devin", help="condition whose first runs seed the committee")
    parser.add_argument("--init-runs", type=int, default=3)
    parser.add_argument("--total", type=int, default=8)
    parser.add_argument("--per-round", type=int, default=2)
    parser.add_argument("--max-probes", type=int, default=400)
    add_backend_args(parser)
    args = parser.parse_args(argv)
    run_active(args.game, args.level, args.train_frac, args.condition, args.init_from, args.init_runs,
               args.total, args.per_round, backend_cfg(args), args.test_level, args.max_probes)


if __name__ == "__main__":
    main()
