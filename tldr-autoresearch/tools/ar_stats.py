#!/usr/bin/env python3
"""
ar_stats.py - summarise an ar_run.py ledger (noise calibration, status counts, per-attempt usage).

  python3 tools/ar_stats.py data/calib_ledger.jsonl [--attempt-prefix calib-] [--json]

For the "ok" runs it reports mean / sample std / min / max of val_bpb, num_steps, training_seconds,
peak_vram_mb, total_tokens_M and wall_s, grouped by train_sha256 (repeats of the same file = noise).
"""
import argparse
import collections
import json
import statistics
import sys

KEYS = ("val_bpb", "num_steps", "training_seconds", "peak_vram_mb", "total_tokens_M", "mfu_percent")


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def describe(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "min": min(vals), "max": max(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--attempt-prefix", default=None, help="only entries whose attempt_id starts with this")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    entries = load(a.ledger)
    if a.attempt_prefix is not None:
        entries = [e for e in entries if (e.get("attempt_id") or "").startswith(a.attempt_prefix)]
    report = {"ledger": a.ledger, "entries": len(entries),
              "status": dict(collections.Counter(e.get("status") for e in entries)),
              "flags": dict(collections.Counter(f for e in entries for f in e.get("flags", []))),
              "by_train_sha256": {}}
    groups = collections.defaultdict(list)
    for e in entries:
        if e.get("status") == "ok":
            groups[e["train_sha256"]].append(e)
    for sha, es in groups.items():
        g = {k: describe([e["metrics"].get(k) for e in es]) for k in KEYS}
        g["wall_s"] = describe([e.get("wall_s") for e in es])
        g["runs"] = [{"run_id": e["run_id"], "attempt_id": e.get("attempt_id"), "val_bpb": e["val_bpb"],
                      "num_steps": e["metrics"].get("num_steps"), "wall_s": e.get("wall_s"),
                      "flags": e.get("flags", [])} for e in es]
        report["by_train_sha256"][sha] = g
    if a.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"{a.ledger}: {report['entries']} entries, status {report['status']}, flags {report['flags']}")
    for sha, g in report["by_train_sha256"].items():
        print(f"\ntrain.py sha256 {sha[:12]}  ({len(g['runs'])} ok runs)")
        for r in g["runs"]:
            print(f"  {r['run_id']}  {r['attempt_id'] or '-':<12} val_bpb {r['val_bpb']:.6f}  "
                  f"steps {r['num_steps']}  wall {r['wall_s']:.1f}s  {','.join(r['flags'])}")
        for k in KEYS + ("wall_s",):
            d = g[k]
            if d:
                print(f"  {k:<17} mean {d['mean']:.6g}  std {d['std']:.4g}  min {d['min']:.6g}  max {d['max']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
