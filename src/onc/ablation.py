"""Single-hypothesis ablations: each template alone against the null, with the committee's machinery.

    uv run python -m onc.ablation --store toy
    uv run python -m onc.ablation --store artifacts/onc/dev --limit 40

A single-template agent is the committee restricted to one hypothesis plus
the null: the same admission, likelihood weights, submission rule and leak
filter, so the difference to the full committee is the committee itself. In
sequential mode every agent here, the committee included, uses the fixed
design of 60 recruited patients with every baseline feature assayed, so the
spend is equal and only the analysis differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from onc.agent import Policy
from onc.evaluate import open_store, run_condition, world_ids
from onc.hypotheses import TEMPLATES

FIXED = {"stop_threshold": 1e9}  # the expected drop never clears it, so the agent submits after the first batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="toy")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="artifacts/onc/ablation")
    args = parser.parse_args(argv)
    store = open_store(args.store)
    rows = []
    for mode in ("full", "seq"):
        ids = world_ids(store, mode, args.limit)
        arms = [("committee, fixed design" if mode == "seq" else "committee", Policy(**FIXED))]
        arms += [(f"single: {name}", Policy(templates=("null", name), **FIXED)) for name in TEMPLATES if name != "null"]
        for label, policy in arms:
            row = run_condition(store, ids, policy, "disagreement")
            row["mode"], row["label"] = mode, f"{mode} / {label}"
            rows.append(row)
            print(f"{row['label']:40s} DS {row['discovery_score'] or float('nan'):.3f} find {row['find'] or float('nan'):.2f} restraint {row['restraint'] or float('nan'):+.2f} cost {row['mean_data_cost']:.0f} ECE drv {row['p_driver'].get('ece', float('nan')):.2f} {row['seconds']:.0f}s", flush=True)
        singles = [r for r in rows if r["mode"] == mode and r["label"].startswith(f"{mode} / single")]
        best = max(singles, key=lambda r: r["discovery_score"])
        print(f"{mode}: best single {best['label']} {best['discovery_score']:.3f}; median single {np.median([r['discovery_score'] for r in singles]):.3f}")
    stem = Path(args.out).with_name(Path(args.out).name + f"_{Path(args.store).name}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps({"store": args.store, "conditions": rows}, indent=1, default=float))
    lines = ["| Condition | DS | 95% | Find | Restraint | Strict | Cost | ECE P(signal) | ECE P(driver) | Members |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        members = float(np.mean([len(w["members"]) for w in r["worlds"]]))
        lines.append(f"| {r['label']} | {r['discovery_score']:.3f} | [{r['interval'][0]:+.2f}, {r['interval'][1]:+.2f}] | {r['find']:.2f} | {r['restraint']:+.2f} | {r['strict']:.2f} | {r['mean_data_cost']:.0f} | {r['p_signal'].get('ece', float('nan')):.2f} | {r['p_driver'].get('ece', float('nan')):.2f} | {members:.1f} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
