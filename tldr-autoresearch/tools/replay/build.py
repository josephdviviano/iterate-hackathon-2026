#!/usr/bin/env python3
"""Build demo/rl_replay.html: an animated, self-contained replay of the online RLTL;DR run (R1).

Reads the run's records (stdlib only, read-only) from --data, by default the exported records in
runs/rl_oct3/ (or $RLTLDR_ROOT/data on the machine that ran it):
  attempts/g*/rollout.json   one per experiment: policy version, idea, result, verdict, the TL;DR insight it
                             produced, and the insights that were in its context
  metrics_trainer.jsonl      one per policy update: optimizer steps, GRPO/SFT tokens, duration
and, when present, <root>/logs/{driver,trainer}.log for the exact run start and publish times (otherwise they
are derived from the rollouts). The records are inlined as JSON into tools/replay/template.html; the page needs
no network. Open it in a browser: space plays/pauses, arrows step, H hides the controls, F goes full screen.

  python3 tools/replay/build.py [--data runs/rl_oct3] [--out demo/rl_replay.html]
"""
import argparse
import datetime as dt
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))


def ts(s):
    return dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def log_times(logs):
    """(run start, {version: publish time}) from the driver/trainer logs, or (None, {}) without them."""
    start, pub = None, {}
    try:
        with open(os.path.join(logs, "driver.log")) as f:
            for line in f:
                if "attempt g0000-a1: policy=" in line:
                    start = ts(line)
                    break
        with open(os.path.join(logs, "trainer.log")) as f:
            for line in f:
                m = re.search(r"published \S+-v(\d+)", line)
                if m:
                    pub[int(m.group(1))] = ts(line)
    except OSError:
        pass
    return start, pub


def main():
    ap = argparse.ArgumentParser()
    root = os.environ.get("RLTLDR_ROOT")
    default_data = os.path.join(root, "data") if root and os.path.isdir(os.path.join(root, "data", "attempts")) \
        else os.path.join(CODE, "runs", "rl_oct3")
    ap.add_argument("--data", default=default_data)
    ap.add_argument("--out", default=os.path.join(CODE, "demo", "rl_replay.html"))
    a = ap.parse_args()
    rolls = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.data, "attempts", "g*", "rollout.json")))]
    rolls = [r for r in rolls if r.get("t_end")]          # an attempt interrupted when the run was stopped
    rolls.sort(key=lambda r: (r["group"], r["k"]))
    start, pub = log_times(os.path.join(os.path.dirname(os.path.abspath(a.data)), "logs"))
    if start is None:   # first experiment's end minus its agent session and a ~6 min training run
        r0 = rolls[0]
        start = r0["t_end"] - ((r0.get("pi") or {}).get("wall_s") or 0) - 360
    exps = []
    for r in rolls:
        oc = r.get("outcome") or {}
        exps.append({
            "id": r["attempt_id"], "g": r["group"], "k": r["k"], "v": (r.get("policy") or {}).get("version", 0),
            "desc": r.get("desc") or "", "val": r.get("val_bpb"), "confirm": r.get("confirm_mean"),
            "status": r.get("status"), "success": bool(r.get("success")),
            "best_before": (r.get("parent") or {}).get("val_bpb"),
            "insight": ((r.get("insight") or {}).get("hint") or None),
            "ctx": [i.get("hint", "") for i in (r.get("insights_in_context") or [])],
            "pi_s": round(((r.get("pi") or {}).get("wall_s") or 0)), "steps": (oc.get("metrics") or {}).get("num_steps"),
            "t_end": round(r["t_end"] - start),
        })
    last_end = {}
    for e in exps:
        last_end[e["g"]] = max(last_end.get(e["g"], 0), e["t_end"])
    updates = []
    for line in open(os.path.join(a.data, "metrics_trainer.jsonl")):
        m = json.loads(line)
        d, v = m.get("data") or {}, m.get("version")
        sec = round(m.get("t_update") or 0)
        t_pub = round(pub[v] - start) if v in pub else (last_end.get(m["group"], 0) + sec)
        updates.append({"g": m["group"], "v": v, "sec": sec, "steps": (m.get("update") or {}).get("steps"),
                        "grpo": d.get("n_grpo_tokens"), "sft": d.get("n_sft_tokens"),
                        "succ": int(sum(d.get("rewards") or [])), "t_pub": t_pub})
    best_final = None
    for e in exps:
        if e["success"]:
            best_final = e["confirm"] or e["val"]
    doc = {"t0": start, "baseline": exps[0]["best_before"], "final_best": best_final, "experiments": exps,
           "updates": updates}
    tpl = open(os.path.join(HERE, "template.html")).read()
    payload = json.dumps(doc, separators=(",", ":")).replace("<", "\\u003c")
    html = tpl.replace("/*__DATA__*/null", payload)
    assert html != tpl, "data placeholder missing"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out}: {len(exps)} experiments, {len(updates)} updates, "
          f"{sum(1 for e in exps if e['insight'])} insights ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
