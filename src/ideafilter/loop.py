"""Round driver for the harness instance: propose, score, decide, write the sweep, keep the ledger.

`plan`: new arms from the proposer (excluding everything in the ledger and every tested
setting), committee scores, the reject rule, one audited rejection, the carried arms from
earlier rounds, and the control, written as a sweep file for the training instance.
`ingest`: a collated table back into the ledger, with outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import importlib.util
import sys
import tomllib

from .backtest import S
from .ledger import add, load, record_measurement, save, write_md
from .liveround import propose, score as score_ideas, toml_value


def lab_config_class():
    path = WT / "research" / "lab_recipe" / "config.py"
    spec = importlib.util.spec_from_file_location("labcfg", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["labcfg"] = mod; spec.loader.exec_module(mod)
    return mod.RecipeConfig


def base_params(adopted: dict, screening: bool) -> dict:
    src = SCREEN_BASE if screening else BASE
    text = "\n".join([l for l in src if l.split(" = ")[0] not in adopted] + [f"{k} = {toml_value(v)}" for k, v in adopted.items()])
    return tomllib.loads(text)


def invalid_reason(levers: dict, adopted: dict, screening: bool, cfg) -> str | None:
    base = base_params(adopted, screening)
    if all(base.get(k) == v for k, v in levers.items()):
        return "duplicate of the base: every lever is already at that value"
    params = dict(base); params.update(levers)
    try:
        cfg.from_parameters(params)
    except Exception as e:  # the lab recipe's own validation message
        return f"invalid under the lab recipe: {e}"
    return None

RULE_THRESHOLD = -0.2
RULE = f"reject if mean predicted accuracy delta <= {RULE_THRESHOLD} pp"
TARGET = 0.752            # the accuracy line the score is read at
SEC_PER_EPOCH = 0.73      # measured on the paired PCIe runs: 6.05 s at 8.25 epochs
EPOCH_STEP = 0.5          # second epoch count per arm in frontier screening
WT = S / "loopwt"
BASE = ["widths = [128, 384, 640]", "block_depth = 3", "translate = 2", "res_schedule = [[0.0, 20], [0.5, 32]]",
        "scaling_factor = 0.16666666666666666", 'global_pool = "flatmax"', "compile = true", "fused_sgd = true",
        "epochs = 8.25", "whiten_grad_off = true"]
SCREEN_BASE = [l for l in BASE if not l.startswith("compile")] + ["compile = false"]  # accuracy screening: no compile, parallel containers
LEAD_PP = 0.10  # a kept arm at or above this, but under 2 SE, is re-run with more seeds


def norm(v):
    v = str(v).replace(" ", "")
    try:
        return str(float(v))
    except ValueError:
        return v


def write_sweep(name: str, arms: list[dict], seeds: list[int], path: Path, adopted: dict, screening: bool = True,
                frontier: bool = False) -> None:
    """With `frontier`, every config (control and arms) is written twice: at the base epochs and EPOCH_STEP fewer."""
    src = SCREEN_BASE if screening else BASE
    base = [l for l in src if l.split(" = ")[0] not in adopted] + [f"{k} = {toml_value(v)}" for k, v in adopted.items()]
    base_epochs = float(dict(l.split(" = ") for l in base)["epochs"])
    lines = [f"# {name}: harness-instance round; arms, decisions and predictions are in the ledger.",
             f'name = "{name}"', 'submission = "research/lab_recipe"', 'python = ".venv-blackwell/bin/python"',
             f"seeds = {seeds}", "devices = [1]", "slots_per_device = 1", "", "[base]", *base, ""]
    epoch_sets = [base_epochs, round(base_epochs - EPOCH_STEP, 2)] if frontier else [None]
    for ep in epoch_sets:
        lines.append("[[configs]]  # control" + (f" at {ep} epochs" if ep else ""))
        if ep: lines.append(f"epochs = {ep}")
    for a in arms:
        for ep in epoch_sets:
            lines.append(f"[[configs]]  # {a['id']} [{a['decision']}]{f' at {ep} epochs' if ep else ''}: {a.get('proposition', '')[:80]}")
            for k, v in a["levers"].items():
                if k != "epochs" or not ep:
                    lines.append(f"{k} = {toml_value(v)}")
            if ep: lines.append(f"epochs = {ep}")
    path.write_text("\n".join(lines) + "\n")


def plan(args) -> None:
    rows = load()
    round_no = max((r["round"] for r in rows), default=0) + 1
    rng = random.Random(round_no)
    adopted = {}
    for r in rows:
        if r["outcome"] == "success" and r["decision"] == "confirm" and r.get("paired_time") and r.get("adopt", True):
            adopted.update(r["levers"])  # adoption needs a confirmed success from a paired, compiled run
    seen = {json.dumps(r["levers"], sort_keys=True) for r in rows}
    ideas = [i for i in propose(args.n, args.proposer_model, ledger_rows=rows, base=base_params(adopted, not args.timing))
             if json.dumps(i["levers"], sort_keys=True) not in seen]
    scored = score_ideas(ideas, args.model)
    cfg = lab_config_class()
    valid = []
    for i in scored:
        reason = invalid_reason(i["levers"], adopted, not args.timing, cfg)
        if reason:
            row = add(rows, round_no, i, "invalid", RULE); row["outcome"] = "invalid"; row["note"] = reason
            print(f"  dropped {i['id']}: {reason}")
        else:
            valid.append(i)
    scored = valid
    kept = [i for i in scored if i["dpp_mean"] > RULE_THRESHOLD]
    rejected = [i for i in scored if i["dpp_mean"] <= RULE_THRESHOLD]
    audit = rng.sample(rejected, min(1, len(rejected)))
    arms = []
    for i in scored:
        decision = "audit" if i in audit else ("run" if i in kept else "rejected")
        row = add(rows, round_no, i, decision, RULE)
        if decision != "rejected":
            arms.append(row)
    # leads: run arms that measured at or above LEAD_PP but under 2 SE get a 20-seed confirmation
    for r in rows:
        if r["decision"] == "run" and r["outcome"] == "failed" and r["dpp"] is not None and r["dpp"] >= LEAD_PP and not r.get("confirmed"):
            row = add(rows, round_no, {"id": r["id"].split("-", 1)[1] + "c", "levers": r["levers"], "proposition": "confirmation: " + r["proposition"],
                                       "p_mean": r["p_mean"], "dpp_mean": r["pred_dpp"], "dtime_mean": r["pred_dtime"], "spread": r["spread"]}, "confirm", RULE)
            row["confirms"] = r["id"]; r["confirmed"] = True
            arms.insert(0, row)
    carried = [r for r in rows if r["decision"] == "carry" and r["outcome"] == "pending"]
    for r in carried:
        r["decision"] = "run"
        arms.append(r)
    order = {"confirm": 0, "audit": 1, "run": 2}
    arms = sorted(arms, key=lambda a: order.get(a["decision"], 3))[: args.max_arms]
    for r in rows:
        if r["decision"] == "run" and r["outcome"] == "pending" and r not in arms:
            r["decision"] = "carry"
    save(rows); write_md(rows)
    name = f"r{round_no}-live"
    path = WT / "research" / "sweeps" / f"{name}.toml"
    seeds = list(range(4000 + 100 * round_no, 4000 + 100 * round_no + args.seeds))
    write_sweep(name, arms, seeds, path, adopted, screening=not args.timing, frontier=args.frontier and not args.timing)
    json.dump({"round": round_no, "rule": RULE, "arms": [a["id"] for a in arms], "audit": [a["id"] for a in arms if a["decision"] == "audit"],
               "rejected": [r["id"] for r in rows if r["round"] == round_no and r["decision"] == "rejected"], "seeds": seeds, "adopted_base": adopted},
              open(S / f"prereg_r{round_no}.json", "w"), indent=1)
    print(f"round {round_no}: {len(scored)} new arms scored, {len(kept)} kept, {len(rejected)} rejected, {len(audit)} audited, "
          f"{len(carried)} carried; sweep {path} with {len(arms)} arms + control, {len(seeds)} seeds; adopted base {adopted}")


def frontier_epochs(points: list[tuple[float, float]]) -> float | None:
    """Epochs at which the line through two (epochs, accuracy) points reaches TARGET; None if flat."""
    (e1, a1), (e2, a2) = sorted(points)
    if a2 == a1:
        return None
    return e1 + (TARGET - a1) * (e2 - e1) / (a2 - a1)


def ingest_frontier(args, rows: list[dict], table: list[dict]) -> None:
    meta = {"config_id", "status", "n", "mean_acc", "sd_acc", "se_acc", "mean_time_local", "mean_time", "error"}
    def levers_of(r): return {k: norm(v) for k, v in r.items() if k not in meta and v and k != "epochs"}
    base_levers = levers_of(table[0])
    def key_of(r): return json.dumps({k: v for k, v in levers_of(r).items() if base_levers.get(k) != v}, sort_keys=True)
    groups: dict = {}
    for r in table:
        groups.setdefault(key_of(r), []).append((float(r["epochs"]), float(r["mean_acc"]), float(r.get("se_acc") or 0)))
    ctrl = groups.get("{}")
    if not ctrl or len(ctrl) < 2:
        print("frontier ingest needs the control at two epoch counts"); return
    e_ctrl = frontier_epochs([(e, a) for e, a, _ in ctrl])
    print(f"control reaches {TARGET:.3f} at {e_ctrl:.2f} epochs" if e_ctrl else "control line is flat")
    matched = 0
    for row in rows:
        if row["outcome"] != "pending" or row["decision"] not in ("run", "audit", "confirm"):
            continue
        if args.round and row["round"] != args.round:
            continue
        want = json.dumps({k: norm(v) for k, v in row["levers"].items() if k != "epochs"}, sort_keys=True)
        pts = groups.get(want)
        if not pts or len(pts) < 2:
            continue
        e_arm = frontier_epochs([(e, a) for e, a, _ in pts])
        hi = max(pts); hi_ctrl = max(ctrl)  # accuracy at the base epochs, for the record
        dpp = (hi[1] - hi_ctrl[1]) * 100
        se = (hi[2] ** 2 + hi_ctrl[2] ** 2) ** 0.5 * 100
        d_epochs = (e_arm - e_ctrl) if (e_arm is not None and e_ctrl is not None) else None
        dtime = (d_epochs * SEC_PER_EPOCH / (e_ctrl * SEC_PER_EPOCH)) if d_epochs is not None else None
        row.update({"dpp": round(dpp, 3), "se": round(se, 3), "n": int(hi[0] and len(pts)), "frontier_epochs": None if e_arm is None else round(e_arm, 3),
                    "control_frontier_epochs": None if e_ctrl is None else round(e_ctrl, 3), "d_epochs": None if d_epochs is None else round(d_epochs, 3),
                    "dtime": None if dtime is None else round(dtime, 4), "paired_time": False, "frontier": True})
        # success: reaches the line at least 0.15 epochs sooner, and its base-epoch accuracy is not below control beyond noise
        row["outcome"] = "success" if (d_epochs is not None and d_epochs <= -0.15 and dpp >= -2 * se) else "failed"
        matched += 1
    save(rows); write_md(rows)
    from collections import Counter
    print(f"frontier: matched {matched} arms; outcomes now {Counter(r['outcome'] for r in rows)}")


def ingest(args) -> None:
    rows = load()
    table = [r for r in csv.DictReader(open(args.table)) if r.get("mean_acc")]
    if args.frontier:
        ingest_frontier(args, rows, table); return
    if args.round:  # re-ingest: reopen that round's measured rows
        for r in rows:
            if r["round"] == args.round and r["decision"] in ("run", "audit", "confirm") and r["dpp"] is not None:
                r["outcome"] = "pending"
    meta = {"config_id", "status", "n", "mean_acc", "sd_acc", "se_acc", "mean_time_local", "mean_time", "error", "host", "block", "arm"}
    ctrl = table[0]
    tcol = "mean_time_local" if "mean_time_local" in ctrl else "mean_time"
    cacc, ct = float(ctrl["mean_acc"]), float(ctrl.get(tcol) or 0)
    base = {k: norm(v) for k, v in ctrl.items() if k not in meta and v}
    matched = 0
    for row in rows:
        if row["outcome"] != "pending" or row["decision"] not in ("run", "audit", "confirm"):
            continue
        if args.round and row["round"] != args.round:
            continue
        want = {k: norm(v) for k, v in row["levers"].items()}
        for r in table[1:]:
            got = {k: norm(v) for k, v in r.items() if k not in meta and v and base.get(k) != norm(v)}
            if got == want:
                t = float(r.get(tcol) or 0)
                record_measurement(row, (float(r["mean_acc"]) - cacc) * 100, ((t - ct) / ct) if (t and ct) else None,
                                   float(r.get("se_acc") or 0) * 100, int(r["n"]), paired=args.paired)
                matched += 1
    save(rows); write_md(rows)
    from collections import Counter
    print(f"matched {matched} arms; outcomes now {Counter(r['outcome'] for r in rows)}")


def plan_confirm(args) -> None:
    """Confirmation sweep: every screening success not yet confirmed, each alone and all stacked, compiled and paired, 20 seeds."""
    rows = load()
    round_no = max((r["round"] for r in rows), default=0) + 1
    cands = [r for r in rows if r["outcome"] == "success" and r["decision"] != "confirm" and not r.get("confirmed")]
    if not cands:
        print("nothing to confirm"); return
    arms = []
    for r in cands:
        row = add(rows, round_no, {"id": r["id"].split("-", 1)[1] + "c", "levers": r["levers"], "proposition": "confirmation: " + r["proposition"],
                                   "p_mean": r["p_mean"], "dpp_mean": r["pred_dpp"], "dtime_mean": r["pred_dtime"], "spread": r["spread"]}, "confirm", RULE)
        row["confirms"] = r["id"]; r["confirmed"] = True; arms.append(row)
    if len(cands) > 1:
        stack = {}
        for r in cands:
            stack.update(r["levers"])
        row = add(rows, round_no, {"id": "stack", "levers": stack, "proposition": "stack of the screening successes: " + ", ".join(r["id"] for r in cands),
                                   "p_mean": None, "dpp_mean": sum(r["pred_dpp"] or 0 for r in cands), "dtime_mean": sum(r["pred_dtime"] or 0 for r in cands), "spread": None}, "confirm", RULE)
        arms.append(row)
    save(rows); write_md(rows)
    name = f"r{round_no}-confirm"
    seeds = list(range(4000 + 100 * round_no, 4000 + 100 * round_no + args.seeds))
    path = WT / "research" / "sweeps" / f"{name}.toml"
    write_sweep(name, arms, seeds, path, {}, screening=False)
    print(f"round {round_no}: confirmation of {[a['id'] for a in arms]} -> {path}, {len(seeds)} seeds, compiled, for the interleave entrypoint")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--n", type=int, default=16); p.add_argument("--max-arms", type=int, default=12)
    p.add_argument("--seeds", type=int, default=10); p.add_argument("--model", default="sonnet"); p.add_argument("--proposer-model", default="opus")
    p.add_argument("--timing", action="store_true", help="write a compiled (max-autotune) sweep for paired timing instead of a screening sweep")
    p.add_argument("--frontier", action="store_true", help="screen every arm at two epoch counts; score = epochs to the 75.2 line")
    p.set_defaults(fn=plan)
    i = sub.add_parser("ingest"); i.add_argument("table"); i.add_argument("--paired", action="store_true", help="time deltas are paired (interleaved run); otherwise accuracy decides")
    i.add_argument("--round", type=int, default=0, help="re-ingest this round's rows")
    i.add_argument("--frontier", action="store_true", help="the table holds every config at two epoch counts"); i.set_defaults(fn=ingest)
    c = sub.add_parser("plan-confirm"); c.add_argument("--seeds", type=int, default=20); c.set_defaults(fn=plan_confirm)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
