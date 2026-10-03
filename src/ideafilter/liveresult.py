"""Score the preregistered live round: rejected arms against kept arms, measured on the sweep.

Reads the sweep's collated table (research/sweep.py collate) and prereg_l1.json; reports the
mean measured accuracy delta and time delta per group, the per-arm predictions beside the
measurements, and whether the preregistered hypothesis held.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

S = Path("/private/tmp/claude-501/-Users-jdv-code-iterate/086a2bad-2d8b-489a-b75f-0f68d03c3e20/scratchpad")
WT = S / "loopwt"


def main(table_csv: str) -> None:
    prereg = json.load(open(S / "prereg_l1.json"))
    rows = [r for r in csv.DictReader(open(table_csv)) if r.get("mean_acc")]
    meta = {"config_id", "status", "n", "mean_acc", "sd_acc", "se_acc", "mean_time_local", "mean_time", "error"}
    lever_cols = [c for c in rows[0] if c not in meta]
    levers = lambda r: {c: r[c] for c in lever_cols if r.get(c) not in ("", None)}
    ctrl = rows[0]  # the sweep lists the control first
    base = levers(ctrl)
    cacc, ct = float(ctrl["mean_acc"]), float(ctrl.get("mean_time_local") or ctrl.get("mean_time") or 0)

    def norm(v):
        v = str(v).replace(" ", "")
        try:
            return str(float(v))
        except ValueError:
            return v

    by_id = {}
    for arm_id, p in prereg["predictions"].items():
        want = {k: norm(v) for k, v in p["levers"].items()}
        for r in rows:
            if r is ctrl: continue
            got = {k: norm(v) for k, v in levers(r).items() if norm(base.get(k)) != norm(v)}
            if got == want:
                t = float(r.get("mean_time_local") or r.get("mean_time") or 0)
                by_id[arm_id] = {"dpp": (float(r["mean_acc"]) - cacc) * 100, "dtime": (t - ct) / ct if ct else None, "status": r["status"], "n": r["n"]}
    groups = {"rejected": prereg["rejected"], "kept": prereg["kept_sample"]}
    print(f"control {cacc*100:.2f}% {ct:.3f}s")
    for g, ids in groups.items():
        print(f"== {g}")
        for i in ids:
            m = by_id.get(i); p = prereg["predictions"][i]
            print(f"  {i:4s} predicted dpp {p['dpp_mean']:+.2f} dtime {p['dtime_mean']:+.3f} | measured " + (f"dpp {m['dpp']:+.2f} dtime {m['dtime']:+.3f} ({m['status']}, n={m['n']})" if m else "missing"))
        vals = [by_id[i]["dpp"] for i in ids if i in by_id]
        if vals: print(f"  mean measured dpp {sum(vals)/len(vals):+.3f} over {len(vals)} arms")
    rj = [by_id[i]["dpp"] for i in groups["rejected"] if i in by_id]; kp = [by_id[i]["dpp"] for i in groups["kept"] if i in by_id]
    if rj and kp:
        diff = sum(kp)/len(kp) - sum(rj)/len(rj)
        print(f"kept minus rejected: {diff:+.3f} pp; preregistered threshold 0.2 pp -> {'HELD' if diff >= 0.2 else 'NOT HELD'}")
    ids = [i for i in prereg["predictions"] if i in by_id]
    pred = [prereg["predictions"][i]["dpp_mean"] for i in ids]; meas = [by_id[i]["dpp"] for i in ids]
    def rank(v):
        order = sorted(range(len(v)), key=lambda k: v[k]); r = [0] * len(v)
        for k, idx in enumerate(order): r[idx] = k
        return r
    rp, rm = rank(pred), rank(meas); n = len(ids); mp, mm = sum(rp) / n, sum(rm) / n
    rho = sum((a - mp) * (b - mm) for a, b in zip(rp, rm)) / (sum((a - mp) ** 2 for a in rp) * sum((b - mm) ** 2 for b in rm)) ** 0.5
    print(f"Spearman(predicted dpp, measured dpp) over {n} arms: {rho:.2f}")
    print("time deltas are not comparable across containers: hosts differ in power limit (400 W vs 500 W SXM); only accuracy is.")


if __name__ == "__main__":
    main(sys.argv[1])
