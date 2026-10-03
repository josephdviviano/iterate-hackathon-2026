"""Collate an interleaved Modal run (research/modal_a100.py::interleave) into a sweep-style table.

Each config ran in every host-block; trials are pooled across blocks, so accuracy and
time are paired with the control on the same host. Writes table.csv next to interleaved.json
with the columns `ideafilter.loop ingest` expects.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


def collate(sweep_root: Path) -> Path:
    rows = json.loads((sweep_root / "interleaved.json").read_text())
    per: dict = {}
    for r in rows:
        cid = r["config_id"]
        d = per.setdefault(cid, {"params": r["params"], "acc": [], "time": [], "blocks": 0, "failed": 0})
        d["blocks"] += 1
        out = sweep_root / "interleaved" / f"h{r['host']}-b{r['block']}-{cid}"
        trials = sorted(out.glob("harness/*/*/trials.jsonl"))
        if not trials or r.get("exit_code"):
            d["failed"] += 1
            continue
        for line in trials[-1].read_text().splitlines():
            t = json.loads(line)
            if t.get("status") == "ok":
                d["acc"].append(t["accuracy"]); d["time"].append(t["total_timed_time"])
    lever_cols = sorted({k for d in per.values() for k in d["params"]})
    out_path = sweep_root / "table.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "status", *lever_cols, "n", "mean_acc", "sd_acc", "se_acc", "mean_time_local", "error"])
        for cid, d in per.items():  # insertion order follows the sweep's config order: control first
            n = len(d["acc"])
            status = "complete" if n and not d["failed"] else "failed"
            sd = statistics.stdev(d["acc"]) if n > 1 else 0.0
            w.writerow([cid, status, *[json.dumps(d["params"][k]) if isinstance(d["params"].get(k), list) else d["params"].get(k, "") for k in lever_cols],
                        n, round(statistics.mean(d["acc"]), 5) if n else "", round(sd, 5), round(sd / n ** 0.5, 5) if n else "",
                        round(statistics.mean(d["time"]), 3) if n else "", "" if status == "complete" else f"{d['failed']} failed blocks"])
    return out_path


if __name__ == "__main__":
    print(collate(Path(sys.argv[1])))
