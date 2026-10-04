#!/usr/bin/env python3
"""Build the run dashboard: collect everything from data/ into one JSON document and inline it into the HTML
template (tools/dashboard/template.html -> dashboard/index.html). Read-only on the run; stdlib only.

    python3 tools/dashboard/build.py [--data DIR] [--out FILE]
"""
import argparse
import glob
import json
import os
import re
import time

# project root: $RLTLDR_ROOT, else the directory above tools/
ROOT = os.environ.get("RLTLDR_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUP_SIZE = 8


def jl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def rj(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return default


def outcome_kind(ro):
    """keep | discard | invalid | crash | norun (what the harness decided, in plain categories)."""
    if ro.get("success"):
        return "keep"
    o = ro.get("outcome")
    if not o:
        return "norun"
    if o.get("status") != "ok":
        return "crash"
    v = " ".join(ro.get("verdict") or [])
    if "not a valid experiment" in v or "could not verify" in v:
        return "invalid"
    return "discard"


def build(data_dir):
    D = data_dir
    st = rj(f"{D}/driver_state.json", {}) or {}
    gw = rj(f"{D}/gateway_state.json", {}) or {}
    ts = rj(f"{D}/trainer_state.json", {}) or {}
    tm = {m["group"]: m for m in jl(f"{D}/metrics_trainer.jsonl")}
    dm = {m["group"]: m for m in jl(f"{D}/metrics_driver.jsonl")}
    baseline, baseline_runs = None, 0

    # ---- experiments (one per rollout.json; voided attempts counted separately)
    exps, n_void = [], 0
    for path in glob.glob(f"{D}/attempts/g*/rollout.json"):
        ro = rj(path)
        if not ro:
            continue
        if ro.get("void"):
            n_void += 1
            continue
        adir = os.path.dirname(path)
        task = os.path.join(adir, "task.md")
        t_start = os.path.getmtime(task) if os.path.exists(task) else None
        o = ro.get("outcome") or {}
        m = o.get("metrics") or {}
        conf = ro.get("confirm") or []
        parent = ro.get("parent") or {}
        kind = outcome_kind(ro)
        exps.append({
            "id": ro["attempt_id"], "group": ro["group"], "k": ro["k"],
            "t_start": t_start, "t_end": ro.get("t_end"),
            "kind": kind, "status": o.get("status"), "flags": o.get("flags") or [],
            "val": o.get("val_bpb") if o.get("status") == "ok" else None,
            "confirm": [c.get("val_bpb") for c in conf],
            "best_before": parent.get("val_bpb"),
            "best_after": ro.get("confirm_mean") if ro.get("success") else parent.get("val_bpb"),
            "policy": (ro.get("policy") or {}).get("version", 0),
            "policy_name": (ro.get("policy") or {}).get("name"),
            "n_insights": len(ro.get("insights_in_context") or []),
            "desc": ro.get("desc") or "",
            "hint": (ro.get("insight") or {}).get("hint"),
            "n_runs": ro.get("n_runs"),
            "agent_s": (ro.get("pi") or {}).get("wall_s"),
            "steps": m.get("num_steps"), "vram_gb": (m.get("peak_vram_mb") or 0) / 1024 if m else None,
        })
    exps.sort(key=lambda e: (e["t_end"] or 0))
    if exps:   # the driver's own baseline: the first experiment's parent (mean of the calibration runs)
        first = rj(f"{D}/attempts/{exps[0]['id']}/rollout.json", {}) or {}
        fp = first.get("parent") or {}
        baseline, baseline_runs = fp.get("val_bpb"), len(fp.get("evals") or [])
    elif st.get("parent"):
        baseline, baseline_runs = st["parent"].get("val_bpb"), len(st["parent"].get("evals") or [])

    # attempt in flight (the gateway clears its attempt during confirmation re-runs and insight writing)
    inflight_id = st.get("inflight")
    gw_att = gw.get("attempt") or {}
    inflight = None
    if inflight_id:
        tmd = f"{D}/attempts/{inflight_id}/task.md"
        inflight = {"id": inflight_id,
                    "since": gw_att.get("t0") if gw_att.get("id") == inflight_id else
                    (os.path.getmtime(tmd) if os.path.exists(tmd) else None),
                    "policy": (gw_att.get("policy") or {}).get("version") if gw_att.get("id") == inflight_id else None,
                    "phase": "agent working" if gw_att.get("id") == inflight_id else "re-run / insight"}
        runs = [e for e in jl(f"{D}/ledger.jsonl") if (e.get("attempt_id") or "").startswith(inflight_id)
                and e.get("status") != "refused"]
        own = [e for e in runs if e.get("attempt_id") == inflight_id]
        conf = [e for e in runs if e.get("attempt_id") != inflight_id]
        inflight.update(
            desc=(own[-1].get("desc") if own else None),
            val=(own[-1].get("val_bpb") if own and own[-1].get("status") == "ok" else None),
            run_status=(own[-1].get("status") if own else None),
            confirm=[c.get("val_bpb") for c in conf],
            n_insights=len(gw_att.get("insights") or []) if gw_att.get("id") == inflight_id else None)
        if inflight["phase"] == "agent working" and own:
            inflight["phase"] = "agent working (run done)"
        elif inflight["phase"] != "agent working" and conf:
            inflight["phase"] = "confirmation re-run / insight"

    # ---- policy versions: trained on group g, published at t, live from the first attempt that used it
    versions = {0: {"version": 0, "name": "base", "group_trained": None, "published": None, "update": None}}
    for g, m in tm.items():
        if m.get("updated") and m.get("version"):
            u = m.get("update") or {}
            d = m.get("data") or {}
            versions[m["version"]] = {
                "version": m["version"], "name": f"v{m['version']}", "group_trained": g, "published": m.get("t"),
                "update": {"grpo_tokens": d.get("n_grpo_tokens"), "sft_tokens": d.get("n_sft_tokens"),
                           "grpo_rollouts": len(d.get("grpo_rollouts") or []), "steps": u.get("steps"),
                           "loss_sft": u.get("loss_sft"), "clipfrac": u.get("clipfrac_by_epoch"),
                           "grad_norm": u.get("grad_norm"), "mismatch": u.get("mismatch_signed"),
                           "t_update": m.get("t_update"), "peak_gib": m.get("peak_mem_gib")},
            }
    for v in versions.values():
        mine = [e for e in exps if e["policy"] == v["version"]]
        v["live_from"] = min((e["t_start"] for e in mine if e["t_start"]), default=None)
        v["live_to"] = max((e["t_end"] for e in mine if e["t_end"]), default=None)
        v["n"] = len(mine)
        v["kept"] = sum(e["kind"] == "keep" for e in mine)
        v["crash"] = sum(e["kind"] in ("crash", "invalid", "norun") for e in mine)
        noins = [e for e in mine if e["n_insights"] == 0]
        v["n_noins"], v["kept_noins"] = len(noins), sum(e["kind"] == "keep" for e in noins)
        v["best_start"] = mine[0]["best_before"] if mine else None
        v["best_end"] = mine[-1]["best_after"] if mine else None
        valid = [e["val"] for e in mine if e["val"] is not None and e["kind"] != "invalid"]
        v["median_val"] = sorted(valid)[len(valid) // 2] if valid else None
        hours = ((v["live_to"] - v["live_from"]) / 3600) if (v["live_from"] and v["live_to"]) else None
        v["hours"] = hours
    # the version that is live right now went live when the attempt in flight started, even before any of its
    # experiments has finished
    pl = (gw.get("policy") or {}).get("version", 0)
    if pl in versions and versions[pl]["live_from"] is None and inflight and inflight["since"]:
        versions[pl]["live_from"] = inflight["since"]
    vlist = sorted(versions.values(), key=lambda v: v["version"])
    for i, v in enumerate(vlist):           # a version is "current" while no later version has gone live
        nxt = next((w for w in vlist[i + 1:] if w["live_from"]), None)
        v["superseded_at"] = nxt["live_from"] if nxt else None

    # ---- groups
    groups = []
    for g in sorted(set(dm) | {e["group"] for e in exps}):
        ge = [e for e in exps if e["group"] == g]
        m = dm.get(g, {})
        t = tm.get(g, {})
        wi = [e for e in ge if e["n_insights"] > 0]
        wo = [e for e in ge if e["n_insights"] == 0]
        rate = lambda xs: (sum(e["kind"] == "keep" for e in xs) / len(xs)) if xs else None
        groups.append({
            "group": g, "closed": g in dm, "n": len(ge), "kept": sum(e["kind"] == "keep" for e in ge),
            "policies": sorted({e["policy"] for e in ge}),
            "succ_noins": m.get("success_rate_no_insight") if m else rate(wo),
            "succ_ins": m.get("success_rate_with_insight") if m else rate(wi),
            "insight_adv": m.get("insight_advantage"), "first": m.get("first_attempt_success"),
            "training": g in dm and g not in tm,
            "best": m.get("best_val_bpb") if m else (ge[-1]["best_after"] if ge else None),
            "update_version": t.get("version") if t.get("updated") else None,
            "updated": t.get("updated"),
            "hints": [{"k": e["k"], "hint": e["hint"]} for e in ge if e["hint"]],
        })

    # ---- headline numbers
    now = time.time()
    t0 = min((e["t_start"] for e in exps if e["t_start"]), default=None)
    starts = sorted([e["t_start"] for e in exps if e["t_start"]] + ([inflight["since"]] if inflight and inflight["since"] else []))
    gaps = [b2 - a2 for a2, b2 in zip(starts[-9:], starts[-8:])]
    cycle = sum(gaps) / len(gaps) if gaps else None            # start-to-start: agent + run + re-run + insight
    train_lags = sorted(tm[g]["t"] - dm[g]["t"] for g in tm if g in dm and tm[g].get("t") and dm[g].get("t"))
    train_lag = train_lags[len(train_lags) // 2] if train_lags else 200.0
    k_done = st.get("k", 0)
    last_closed = max(dm) if dm else None
    if gw.get("pending"):
        eta, eta_note = None, "published; goes live at the next experiment"
    elif last_closed is not None and last_closed not in tm:
        eta, eta_note = dm[last_closed]["t"] + train_lag, f"training on group {last_closed}"
    elif cycle:
        base_t = inflight["since"] if inflight and inflight["since"] else now
        eta, eta_note = base_t + (GROUP_SIZE - k_done) * cycle + train_lag, None
    else:
        eta, eta_note = None, None
    best = (st.get("parent") or {}).get("val_bpb")
    # insights actually injected into the experiment in flight (only while the group's success rate <= 50%)
    in_play = gw_att.get("insights") if gw_att.get("id") else None
    summary = {
        "generated_at": now, "run_started": t0, "baseline": baseline,
        "best": best, "best_commit": ((st.get("parent") or {}).get("commit") or "")[:7],
        "n_exps": len(exps), "n_kept": sum(e["kind"] == "keep" for e in exps), "n_void": n_void,
        "baseline_runs": baseline_runs,
        "group": st.get("group"), "k_done": k_done, "in_progress": inflight,
        "policy_live": (gw.get("policy") or {}).get("version", 0),
        "policy_pending": (gw.get("pending") or {}).get("version"),
        "trainer_version": ts.get("version", 0),
        "per_hour": 3600 / cycle if cycle else None,
        "avg_exp_min": cycle / 60 if cycle else None,
        "next_update_eta": eta, "next_update_note": eta_note,
        "current_insights": in_play if in_play is not None else (st.get("insights") or []),
        "insights_injected": in_play is not None,
    }
    return {"summary": summary, "experiments": exps, "versions": vlist, "groups": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data")
    ap.add_argument("--out", default=f"{ROOT}/dashboard/index.html")
    a = ap.parse_args()
    doc = build(a.data)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")).read()
    payload = json.dumps(doc, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/{}", payload)
    assert html != tpl, "data placeholder missing from template"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    open(tmp, "w").write(html)
    os.replace(tmp, a.out)
    s = doc["summary"]
    print(f"built {a.out}: {s['n_exps']} experiments, {s['n_kept']} kept, best {s['best']}, "
          f"policy live v{s['policy_live']}, {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
