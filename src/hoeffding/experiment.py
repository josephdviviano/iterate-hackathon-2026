"""Baselines, synthesis runs and the report for Hoeffding's problem.

Layout: artifacts/hoeffding/
  baselines.json          bernoulli, hoeffding wall, family best, search best, reference per instance
  synth/<seed>/run<k>/    strategy.py, report.json, meta.json
  report.json             committee report on test instances, calibration
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

from .committee import HoeffdingCommittee, Member, run_strategy
from .families import exact_n1_cert, family_best
from .problem import TEST, TRAIN, Instance, bernoulli_value, hoeffding_bound
from .search import search_best
from .synth import SEEDS, synthesize

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "hoeffding"
TOL = 1e-4


def baselines(instances: list[Instance], restarts: int = 20) -> dict:
    out = {}
    for inst in instances:
        t0 = time.time()
        fam, fam_name = family_best(inst)
        t_fam = time.time() - t0
        t0 = time.time()
        srch, k = search_best(inst, restarts=restarts)
        t_srch = time.time() - t0
        t0 = time.time()
        fam_recheck = fam.value  # certificate already exact; time one certification call for the record
        from .verify import tail_probability
        tail_probability(fam.atoms, fam.weights, inst.n, inst.t)
        t_cert = time.time() - t0
        if inst.n == 1:
            ref, ref_kind = exact_n1_cert(inst).value, "exact (Markov)"
        elif inst.n == 2:
            ref, ref_kind = fam.value, "exact (Meester 2008)"
        else:
            ref, ref_kind = max(fam.value, srch.value if srch else Fraction(0)), "best known here, not proven"
        out[inst.key] = {
            "instance": inst.to_json(),
            "bernoulli": float(bernoulli_value(inst)),
            "hoeffding_upper": hoeffding_bound(inst),
            "family_best": fam.to_json() | {"family": fam_name, "seconds": round(t_fam, 3)},
            "search_best": (srch.to_json() | {"k": k, "seconds": round(t_srch, 3)}) if srch else None,
            "certificate_seconds": round(t_cert, 4),
            "reference": {"value": float(ref), "kind": ref_kind},
        }
        print(f"{inst.key:24s} bern {out[inst.key]['bernoulli']:.6f}  family {float(fam.value):.6f} ({fam_name})  "
              f"search {float(srch.value) if srch else float('nan'):.6f}  wall {hoeffding_bound(inst):.6f}  "
              f"cert {t_cert*1000:.1f} ms")
    return out


def cmd_baselines(args):
    ART.mkdir(parents=True, exist_ok=True)
    res = {"train": baselines(TRAIN, args.restarts), "test": baselines(TEST, args.restarts)}
    (ART / "baselines.json").write_text(json.dumps(res, indent=1))


def cmd_synth(args):
    for seed_name in args.seeds:
        for k in range(args.runs):
            out = ART / "synth" / seed_name / f"run{k}"
            if (out / "strategy.py").exists() and not args.overwrite:
                continue
            out.mkdir(parents=True, exist_ok=True)
            r = synthesize(TRAIN, SEEDS[seed_name], model=args.model, max_turns=args.max_turns,
                           timeout_s=args.timeout)
            (out / "strategy.py").write_text(r.source)
            (out / "report.json").write_text(json.dumps(r.report, indent=1))
            (out / "meta.json").write_text(json.dumps(dict(r.meta, seed=seed_name, check_output=r.check_output),
                                                      indent=1))
            print(f"{seed_name} run{k}: {r.meta.get('num_turns')} turns, {r.meta.get('wall_s')} s, "
                  f"${r.meta.get('total_cost_usd')}")


def cmd_report(args):
    base = json.loads((ART / "baselines.json").read_text())
    out = {}
    for split, instances in (("train", TRAIN), ("test", TEST)):
        members = []
        for p in sorted((ART / "synth").glob("*/run*/strategy.py")):
            name = f"{p.parent.parent.name}/{p.parent.name}"
            src = p.read_text()
            rep = json.loads((p.parent / "report.json").read_text()) if (p.parent / "report.json").exists() else {}
            confs: list = []
            results = run_strategy(src, instances, confidences=confs)
            p_tight = [c if c is not None else (rep.get(i.key) if split == "train" else None)
                       for c, i in zip(confs, instances)]
            members.append(Member(name, src, results, p_tight))
            n_ok = sum(not isinstance(r, str) for r in results)
            print(f"{split} {name}: {n_ok}/{len(instances)} instances certified")
        if not members:
            print("no synthesized members; report covers baselines only")
        rows = build_rows(base[split], instances, members)
        cal = calibration(rows)
        out[split] = {"rows": rows, "calibration": cal}
        print(f"\n== {split}")
        print_table(rows, cal)
    (ART / "report.json").write_text(json.dumps(out, indent=1))


def build_rows(base_split, instances, members):
    rows = []
    for i, inst in enumerate(instances):
        b = base_split[inst.key]
        ref = b["reference"]["value"]
        row = {"key": inst.key, "n": inst.n, "bernoulli": b["bernoulli"], "family": b["family_best"]["value_float"],
               "search": b["search_best"]["value_float"] if b["search_best"] else None,
               "hoeffding_upper": b["hoeffding_upper"], "reference": ref, "reference_kind": b["reference"]["kind"]}
        if members:
            com = HoeffdingCommittee(members)
            rep = com.report(i, inst)
            row.update({"committee_lower": rep["certified_lower"], "disagreement": rep["disagreement"],
                        "leader": rep["leader_distribution"], "structures": rep["member_structures"]})
            for m in members:
                v = m.value(i)
                pt = m.p_tight[i]
                row.setdefault("members", {})[m.name] = {
                    "value": float(v) if v is not None else None,
                    "tight": (v is not None and ref - float(v) <= TOL),
                    "p_tight": pt,
                }
        rows.append(row)
    return rows


def calibration(rows: list[dict]) -> dict:
    """Brier score of each member's p_tight against whether its value reached the reference, by reference kind."""
    out: dict = {}
    for row in rows:
        for name, mm in row.get("members", {}).items():
            if mm["p_tight"] is None:
                continue
            kind = "exact" if row["reference_kind"].startswith("exact") else "reference"
            d = out.setdefault(name, {}).setdefault(kind, {"n": 0, "brier": 0.0, "mean_p": 0.0, "hit_rate": 0.0})
            d["n"] += 1
            d["brier"] += (mm["p_tight"] - mm["tight"]) ** 2
            d["mean_p"] += mm["p_tight"]
            d["hit_rate"] += mm["tight"]
    for name in out:
        for kind, d in out[name].items():
            for k in ("brier", "mean_p", "hit_rate"):
                d[k] = round(d[k] / d["n"], 4)
    return out


def print_table(rows, cal):
    print(f"\n{'key':24s} {'bern':>8s} {'family':>8s} {'search':>8s} {'commit':>8s} {'ref':>8s} {'wall':>8s} {'unknown':>8s}  ref kind")
    for r in rows:
        c = r.get("committee_lower")
        lower = max(x for x in (r["family"], r["search"] or 0, c or 0))
        print(f"{r['key']:24s} {r['bernoulli']:8.5f} {r['family']:8.5f} {(r['search'] or float('nan')):8.5f} "
              f"{(c if c is not None else float('nan')):8.5f} {r['reference']:8.5f} {r['hoeffding_upper']:8.5f} "
              f"{r['hoeffding_upper']-lower:8.5f}  {r['reference_kind']}")
    if cal:
        print("\ncalibration (Brier of p_tight vs tight):")
        for name, kinds in cal.items():
            for kind, d in kinds.items():
                print(f"  {name:20s} {kind:10s} n={d['n']:2d} brier={d['brier']:.3f} mean_p={d['mean_p']:.2f} hit={d['hit_rate']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines"); b.add_argument("--restarts", type=int, default=20); b.set_defaults(fn=cmd_baselines)
    s = sub.add_parser("synth")
    s.add_argument("--seeds", nargs="+", default=list(SEEDS)); s.add_argument("--runs", type=int, default=1)
    s.add_argument("--model", default="opus"); s.add_argument("--max-turns", type=int, default=40)
    s.add_argument("--timeout", type=float, default=900); s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_synth)
    r = sub.add_parser("report"); r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
