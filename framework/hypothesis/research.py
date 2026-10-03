"""
research.py: bookkeeping for the hypothesis-driven research framework. Task-agnostic.

It keeps the research state (hypotheses, ideas, world model, one notebook page per experiment)
and links it to experiments. Running and measuring experiments is NOT its job: it calls
`ar.py start` / `ar.py log`, exactly like any other framework, so the measurement is the task's.

State (workspace root, untracked by git):
    hypothesis.tsv  hypotheses, Lvl1 (broad claim) > Lvl2 (mechanism) > Lvl3 (falsifiable prediction)
    ideas.tsv       queue of experiment ideas, each testing one hypothesis
    predictions.tsv per experiment: idea, hypothesis, verdict, prediction check, pre-registration
    meta.tsv        journal of the meta-steps (ideate / revise)
    world_model.md  current understanding; read before and updated after every experiment
    notebook/       E007.md: hypothesis chain, pre-registration, contingencies, outcome, post-mortem
    history/        world model snapshot after every experiment
    drafts/         the next experiment, prepared while the current one runs

    python research.py init
    python research.py status
    python research.py hypo add|set|tree ...      python research.py idea add|next|set|list ...
    python research.py launch --idea I004 [--prereg drafts/I004.prereg.md]   (or --idea - --description ...)
    python research.py prereg                     # seal the pre-registration of the running experiment
    python research.py log --status keep|discard|crash --verdict supports|refutes|inconclusive|-
    python research.py meta ideate|revise --summary "..."
    python research.py snapshot
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
AR = [sys.executable, os.path.join(ROOT, "ar.py")]
TASK_FILE = os.path.join(ROOT, "task.json")
RESULTS_FILE = os.path.join(ROOT, "results.tsv")
INFLIGHT_FILE = os.path.join(ROOT, ".ar", "inflight.json")
RUNS_DIR = os.path.join(ROOT, "runs")
HYPO_FILE = os.path.join(ROOT, "hypothesis.tsv")
IDEAS_FILE = os.path.join(ROOT, "ideas.tsv")
PRED_FILE = os.path.join(ROOT, "predictions.tsv")
META_FILE = os.path.join(ROOT, "meta.tsv")
WORLD_MODEL_FILE = os.path.join(ROOT, "world_model.md")
NOTEBOOK_DIR = os.path.join(ROOT, "notebook")
HISTORY_DIR = os.path.join(ROOT, "history")
DRAFTS_DIR = os.path.join(ROOT, "drafts")
SEAL_FILE = os.path.join(ROOT, ".ar", "seal.json")
STATE = ["hypothesis.tsv", "ideas.tsv", "predictions.tsv", "meta.tsv", "world_model.md",
         "notebook/", "history/", "drafts/"]  # fmt: skip

HYPO_COLS = ["id", "level", "parent", "status", "confidence", "evidence", "source", "statement"]
IDEA_COLS = ["id", "hypothesis", "priority", "status", "expected", "description", "note"]
PRED_COLS = ["exp", "idea", "hypothesis", "status", "verdict", "pred", "prereg"]
META_COLS = ["step", "kind", "after_exp", "summary"]
HYPO_STATUSES = ["open", "supported", "refuted", "inconclusive", "retired"]
CLOSED = ("refuted", "retired")
IDEA_STATUSES = ["queued", "running", "done", "dropped"]
VERDICTS = {"supports": "+", "refutes": "-", "inconclusive": "~", "-": ""}
IDEATE_MIN, IDEATE_REFILL, REVISE_EVERY, PLATEAU = 3, 5, 10, 5

WORLD_MODEL = """\
# World model

_Last updated after: (none yet)_

## Current best
<!-- the current recipe in a line or two, and its metrics -->

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->

## Where the cost goes
<!-- what dominates the objective, and why -->

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->

## Current beliefs
<!-- the Lvl1/Lvl2 hypotheses that drive the search now, and why -->

## Lessons from failures
<!-- distilled from post-mortems: what the model got wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
"""

PAGE = """\
# {exp} · {idea} · {hypothesis}

{chain}
Idea: {description}
Expected (from the queue): {expected}
Commit {commit}; tip at launch: {tip}; run log: runs/{exp}.log

## Pre-registration
<!-- Before you look at this run's output. One prediction line per metric you predict, as a range:
`- Prediction: <metric> [lo, hi]` with a metric name from task.json. -->
- Prediction:
- Supports if:
- Refutes if:

## Contingencies
<!-- The next step for each outcome, decided now. -->
- If it lands in the predicted range:
- If it moves the right way but less than predicted:
- If it moves the opposite way:
- If it crashes:

## In-flight notes

## Outcome
<!-- After the run: the numbers, which contingency fired, the verdict and why. -->

## Post-mortem
<!-- For failures and missed predictions: what you predicted, what happened, the root cause,
what the world model got wrong, the lesson. -->
"""

RANGE = r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def read_text(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_json(path, default=None):
    return json.loads(read_text(path)) if os.path.exists(path) else default


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def clean(v):
    return re.sub(r"[\t\r\n]+", " ", str(v)).strip()


def read_tsv(path, cols):
    if not os.path.exists(path):
        die(f"{os.path.basename(path)} not found; run `python research.py init` first")
    lines = [line for line in read_text(path).splitlines() if line.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t") + [""] * len(header))) for line in lines[1:]]


def write_tsv(path, cols, rows):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(clean(r.get(c, "")) for c in cols) + "\n")
    os.replace(path + ".tmp", path)


def find(rows, rid, what):
    for r in rows:
        if r["id"] == rid:
            return r
    die(f"no {what} {rid}")


def ar(*args):
    """Run ar.py and echo its output; return (exit code, stdout)."""
    p = subprocess.run([*AR, *args], cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    return p.returncode, p.stdout


def results():
    if not os.path.exists(RESULTS_FILE):
        return []
    lines = [line for line in read_text(RESULTS_FILE).splitlines() if line.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t") + [""] * len(header))) for line in lines[1:]]


def last_exp():
    rows = results()
    return rows[-1]["exp"] if rows else "-"


def exp_num(e):
    return int(e[1:]) if e and e[1:].isdigit() else -1


def tip():
    for r in reversed(results()):
        if r["status"] == "keep":
            return r
    return None


def chain(hypos, hid):
    by_id, out = {h["id"]: h for h in hypos}, []
    while hid in by_id:
        out.append(by_id[hid])
        hid = by_id[hid]["parent"]
    return out


def closed_in_chain(hypos, hid):
    for h in chain(hypos, hid):
        if h["status"] in CLOSED:
            return h["id"]
    return None


def section(text, title):
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text.replace("\r\n", "\n"),
                  re.MULTILINE | re.DOTALL)  # fmt: skip
    return re.sub(r"<!--.*?-->", "", m.group(1), flags=re.DOTALL).strip() if m else None


def block(text, title):
    m = re.search(rf"^## {re.escape(title)}\s*$.*?(?=^## |\Z)", text.replace("\r\n", "\n"),
                  re.MULTILINE | re.DOTALL)  # fmt: skip
    return m.group(0) if m else ""


def page_path(exp):
    return os.path.join(NOTEBOOK_DIR, f"{exp}.md")


def snapshot_path(exp):
    return os.path.join(HISTORY_DIR, f"world_model_{exp}.md")


def metric_names():
    return list((read_json(TASK_FILE, {}) or {}).get("metrics", {}))


def predictions(text):
    """[(metric, lo, hi)] from `- Prediction: <metric> [lo, hi]` lines of the pre-registration."""
    out = []
    for line in (section(text, "Pre-registration") or "").splitlines():
        m = re.match(rf"^- Prediction:\s*([A-Za-z_][\w.]*)\s*{RANGE}", line.strip())
        if m:
            lo, hi = sorted((float(m.group(2)), float(m.group(3))))
            out.append((m.group(1), lo, hi))
    return out


def prereg_problems(text):
    problems = []
    preds = predictions(text)
    if not preds:
        problems.append("no `- Prediction: <metric> [lo, hi]` line")
    unknown = [p[0] for p in preds if p[0] not in metric_names()]
    if unknown:
        problems.append(f"unknown metric(s) {unknown}; task.json has {metric_names()}")
    body = section(text, "Pre-registration") or ""
    for field in ("Supports if", "Refutes if"):
        m = re.search(rf"^- {field}:(.*)$", body, re.MULTILINE)
        if not m or not m.group(1).strip():
            problems.append(f"'{field}' is empty")
    filled = [ln for ln in (section(text, "Contingencies") or "").splitlines() if re.match(r"^- .+:\s*\S", ln)]  # fmt: skip
    if len(filled) < 3:
        problems.append(f"{len(filled)} contingencies filled (need 3)")
    return problems


# ---------------------------------------------------------------------------
# triggers
# ---------------------------------------------------------------------------


def missing_postmortems(preds):
    out = []
    for p in preds:
        failed = p["status"] in ("discard", "crash") or p["verdict"] == "refutes" or "miss" in p["pred"]
        if p["idea"] != "-" and failed and not section(read_text(page_path(p["exp"])), "Post-mortem"):
            out.append(p["exp"])
    return out


def due(hypos, ideas, preds, metas):
    """(before the next launch, agenda for the next run's downtime)"""
    blocking, agenda = [], []
    if results() and not os.path.exists(snapshot_path(last_exp())):
        blocking.append(f"world model: update world_model.md for {last_exp()}, then `snapshot`")
    queued = [i for i in ideas if i["status"] == "queued"]
    if not queued and not any(i["status"] == "running" for i in ideas):
        (blocking if results() else agenda).append("ideate: the queue is empty")
    elif len(queued) < IDEATE_MIN:
        agenda.append(f"ideate: {len(queued)} idea(s) queued; refill to >= {IDEATE_REFILL}")
    pm = missing_postmortems(preds)
    if pm:
        agenda.insert(0, f"post-mortems: {', '.join(pm)}")
    reasons = []
    if not [h for h in hypos if h["status"] == "open" and not closed_in_chain(hypos, h["id"])]:
        reasons.append("no open hypotheses")
    revises = [m for m in metas if m["kind"] == "revise"]
    last = exp_num(revises[-1]["after_exp"]) if revises else -1
    if results() and exp_num(last_exp()) - last >= REVISE_EVERY:
        reasons.append(f"{exp_num(last_exp()) - last} experiments since the last revise")
    streak = 0
    for r in reversed(results()):
        if r["status"] == "keep" or exp_num(r["exp"]) <= last:
            break
        streak += 1
    if streak >= PLATEAU:
        reasons.append(f"{streak} experiments without a keep")
    if reasons:
        agenda.insert(1 if pm else 0, "revise: " + "; ".join(reasons))
    return blocking, agenda


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_init(args):
    if any(os.path.exists(os.path.join(ROOT, s)) for s in STATE) and not args.force:
        die("research state already exists")
    write_tsv(HYPO_FILE, HYPO_COLS, [])
    write_tsv(IDEAS_FILE, IDEA_COLS, [])
    write_tsv(PRED_FILE, PRED_COLS, [])
    write_tsv(META_FILE, META_COLS, [])
    with open(WORLD_MODEL_FILE, "w") as f:
        f.write(WORLD_MODEL)
    for d in (NOTEBOOK_DIR, HISTORY_DIR, DRAFTS_DIR):
        os.makedirs(d, exist_ok=True)
    exclude = os.path.join(ROOT, ".git", "info", "exclude")
    have = read_text(exclude).splitlines()
    with open(exclude, "a") as f:
        for s in STATE:
            if "/" + s not in have:
                f.write("/" + s + "\n")
    print("initialized the research state")


def cmd_status(args):
    hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
    preds, metas = read_tsv(PRED_FILE, PRED_COLS), read_tsv(META_FILE, META_COLS)
    ar("status")
    honest = [p for p in preds if p["pred"] and p["prereg"] == "blind"][-10:]
    if honest:
        for metric in metric_names():
            marks = re.findall(rf"\b{re.escape(metric)}:(hit|miss)", " ".join(p["pred"] for p in honest))
            if marks:
                print(f"prediction calibration, {metric}: {marks.count('hit')}/{len(marks)} hit (last 10 blind)")
    print("\nopen hypotheses (`hypo tree` shows all):")
    print_tree(hypos, only_open=True)
    queue = sorted([i for i in ideas if i["status"] in ("queued", "running")],
                   key=lambda i: (i["status"] != "running", -int(i["priority"] or 0), i["id"]))  # fmt: skip
    print(f"\nqueue ({sum(i['status'] == 'queued' for i in ideas)} queued):")
    for i in queue[:8]:
        print(f"  {i['id']} p{i['priority']} {i['status']:<8} {i['hypothesis']:<8} {i['description']}  -> {i['expected']}")
    blocking, agenda = due(hypos, ideas, preds, metas)
    fl = read_json(INFLIGHT_FILE)
    if fl:
        seal = read_json(SEAL_FILE, {})
        print(f"\nwhile {fl['exp']} runs, in this order:")
        steps = [] if seal.get("exp") == fl["exp"] else [f"pre-register {fl['exp']} in notebook/{fl['exp']}.md, then `prereg`"]
        if results():
            steps.append(f"world model: finish the full update for {last_exp()}, then `snapshot` again")
        steps += agenda + ["draft the next idea (patch + pre-registration) in drafts/"]
        for n, s in enumerate(steps, 1):
            print(f"  {n}. {s}")
    else:
        print("\ndue before the next launch:" if blocking else "\nnothing blocks the next launch")
        for b in blocking:
            print(f"  {b}")
        if agenda:
            print("for the next run's downtime: " + "; ".join(agenda))


def print_tree(hypos, only_open=False):
    kids = {}
    for h in hypos:
        kids.setdefault(h["parent"], []).append(h)

    def walk(parent, depth):
        for h in kids.get(parent, []):
            if only_open and h["status"] in CLOSED:
                continue
            ev = f"  [{h['evidence']}]" if h["evidence"] else ""
            print(f"{'  ' * depth}{h['id']:<8} {h['status']:<12} c={h['confidence']:<4} {h['statement']}{ev}")
            walk(h["id"], depth + 1)

    walk("", 0)


def cmd_hypo_add(args):
    hypos = read_tsv(HYPO_FILE, HYPO_COLS)
    parent = args.parent or ""
    if parent:
        find(hypos, parent, "hypothesis")
        if closed_in_chain(hypos, parent):
            die(f"{closed_in_chain(hypos, parent)} is closed; add under a live hypothesis")
        if parent.count(".") >= 2:
            die(f"{parent} is already level 3")
    prefix = f"{parent}." if parent else "H"
    taken = [int(h["id"][len(prefix):]) for h in hypos
             if h["id"].startswith(prefix) and h["id"][len(prefix):].isdigit()]  # fmt: skip
    hid = f"{prefix}{max(taken, default=0) + 1}"
    hypos.append({"id": hid, "level": str(hid.count(".") + 1), "parent": parent, "status": "open",
                  "confidence": f"{float(args.confidence):.2f}", "evidence": "",
                  "source": args.source, "statement": args.statement})  # fmt: skip
    write_tsv(HYPO_FILE, HYPO_COLS, hypos)
    print(hid)


def cmd_hypo_set(args):
    hypos = read_tsv(HYPO_FILE, HYPO_COLS)
    h = find(hypos, args.id, "hypothesis")
    for field in ("status", "statement", "source"):
        if getattr(args, field):
            h[field] = getattr(args, field)
    if args.confidence is not None:
        h["confidence"] = f"{float(args.confidence):.2f}"
    if args.evidence:
        h["evidence"] = " ".join(filter(None, [h["evidence"], args.evidence]))
    msg = f"{h['id']} {h['status']} c={h['confidence']}"
    if h["status"] in CLOSED:  # a closed hypothesis takes its open subtree and queued ideas with it
        sub = [d for d in hypos if d["id"].startswith(h["id"] + ".")]
        for d in sub:
            if d["status"] == "open":
                d["status"] = "retired"
        ids = {h["id"]} | {d["id"] for d in sub}
        ideas = read_tsv(IDEAS_FILE, IDEA_COLS)
        dropped = [i["id"] for i in ideas if i["status"] == "queued" and i["hypothesis"] in ids]
        for i in ideas:
            if i["id"] in dropped:
                i["status"], i["note"] = "dropped", f"{h['id']} {h['status']}"
        write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
        msg += f"; retired the open subtree, dropped {', '.join(dropped) or 'no ideas'}"
    write_tsv(HYPO_FILE, HYPO_COLS, hypos)
    print(msg)


def cmd_hypo_tree(args):
    print_tree(read_tsv(HYPO_FILE, HYPO_COLS))


def cmd_idea_add(args):
    hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
    find(hypos, args.hypothesis, "hypothesis")
    if closed_in_chain(hypos, args.hypothesis):
        die(f"{closed_in_chain(hypos, args.hypothesis)} is closed")
    nums = [int(i["id"][1:]) for i in ideas if i["id"][1:].isdigit()]
    iid = f"I{max(nums, default=0) + 1:03d}"
    ideas.append({"id": iid, "hypothesis": args.hypothesis, "priority": str(args.priority),
                  "status": "queued", "expected": args.expected, "description": args.description,
                  "note": ""})  # fmt: skip
    write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
    print(iid)


def cmd_idea_next(args):
    hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
    running = [i for i in ideas if i["status"] == "running"]
    if running:
        i = running[0]
    else:
        queued = [i for i in ideas if i["status"] == "queued" and not closed_in_chain(hypos, i["hypothesis"])]
        if not queued:
            die("the queue is empty: run the ideate meta-step")
        i = sorted(queued, key=lambda i: (-int(i["priority"] or 0), i["id"]))[0]
        if not args.peek:
            i["status"] = "running"
            write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
    print(f"{i['id']} ({i['status']}): {i['description']}\nexpected: {i['expected']}")
    for h in chain(hypos, i["hypothesis"]):
        print(f"  {h['id']} {h['status']} c={h['confidence']}: {h['statement']}")
    drafts = [f for f in sorted(os.listdir(DRAFTS_DIR)) if f.startswith(i["id"])] if os.path.isdir(DRAFTS_DIR) else []
    if drafts:
        print("drafts: " + ", ".join("drafts/" + d for d in drafts))


def cmd_idea_set(args):
    ideas = read_tsv(IDEAS_FILE, IDEA_COLS)
    i = find(ideas, args.id, "idea")
    for field in ("status", "note", "expected", "description"):
        if getattr(args, field):
            i[field] = getattr(args, field)
    if args.priority is not None:
        i["priority"] = str(args.priority)
    write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
    print(f"{i['id']} {i['status']} p{i['priority']}")


def cmd_idea_list(args):
    for i in read_tsv(IDEAS_FILE, IDEA_COLS):
        if args.all or i["status"] in ("queued", "running"):
            print(f"{i['id']} p{i['priority']} {i['status']:<8} {i['hypothesis']:<8} {i['description']}  -> {i['expected']}"
                  + (f"  ({i['note']})" if i["note"] else ""))  # fmt: skip


def seal(text, exp, at_launch=False):
    fl = read_json(INFLIGHT_FILE) or {}
    log_text = read_text(fl.get("log"))
    task = read_json(TASK_FILE, {})
    seen = any(re.search(s["regex"], log_text, re.MULTILINE) for s in task.get("metrics", {}).values())
    write_json(SEAL_FILE, {"exp": exp, "at": time.time(), "text": block(text, "Pre-registration")
                           + block(text, "Contingencies"), "late": bool(seen) and not at_launch})  # fmt: skip


def cmd_launch(args):
    hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
    preds, metas = read_tsv(PRED_FILE, PRED_COLS), read_tsv(META_FILE, META_COLS)
    blocking, _ = due(hypos, ideas, preds, metas)
    if args.idea == "-":
        blocking = [b for b in blocking if not b.startswith("ideate")]
    if blocking:
        die("before the next launch:\n  " + "\n  ".join(blocking))
    if args.idea == "-":
        if not args.description:
            die("--description is required with --idea -")
        idea, hid = None, "-"
    else:
        idea = find(ideas, args.idea, "idea")
        if idea["status"] != "running":
            die(f"{idea['id']} is {idea['status']}; pop it with `idea next` first")
        hid = idea["hypothesis"]
    draft = read_text(args.prereg) if args.prereg else ""
    if args.prereg and prereg_problems(draft):
        die(f"{args.prereg}: " + "; ".join(prereg_problems(draft)))
    desc = args.description or idea["description"]
    code, _ = ar("start", "--tag", idea["id"] if idea else "-", "--description", desc)
    if code != 0:
        sys.exit(code)
    fl = read_json(INFLIGHT_FILE)
    t = tip()
    page = PAGE.format(
        exp=fl["exp"], idea=idea["id"] if idea else "-", hypothesis=hid,
        chain="\n".join(f"{h['id']} ({h['status']}, c={h['confidence']}): {h['statement']}"
                        for h in chain(hypos, hid)) or "(no hypothesis)",
        description=desc, expected=idea["expected"] if idea else "-", commit=fl["commit"],
        tip=f"{t['exp']} @ {t['commit']}" if t else "none",
    )  # fmt: skip
    if draft:
        for title in ("Pre-registration", "Contingencies"):
            page = page.replace(block(page, title), block(draft, title).rstrip("\n") + "\n\n")
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    with open(page_path(fl["exp"]), "w") as f:
        f.write(page)
    if draft:
        seal(page, fl["exp"], at_launch=True)
        print("pre-registration sealed at launch")
    else:
        print(f"now, before looking at its output: fill the pre-registration in notebook/{fl['exp']}.md, then `prereg`")


def cmd_prereg(args):
    fl = read_json(INFLIGHT_FILE)
    if not fl:
        die("no experiment is running")
    if read_json(SEAL_FILE, {}).get("exp") == fl["exp"]:
        die("already sealed")
    text = read_text(page_path(fl["exp"]))
    if prereg_problems(text):
        die("; ".join(prereg_problems(text)))
    seal(text, fl["exp"])
    late = read_json(SEAL_FILE)["late"]
    print(f"sealed {fl['exp']}" + (" (late: results were already in the log)" if late else " (blind)"))


def cmd_log(args):
    fl = read_json(INFLIGHT_FILE)
    if not fl:
        die("no experiment is running")
    passthrough = ["log", "--status", args.status] + (["--description", args.description] if args.description else [])
    code, _ = ar(*passthrough, *(["--force"] if args.force else []))
    if code != 0:
        sys.exit(code)
    row = results()[-1]
    run = read_json(os.path.join(RUNS_DIR, f"{row['exp']}.json"), {})
    metrics = run.get("metrics", {})
    status, verdict = row["status"], args.verdict
    if status == "crash" and verdict != "-":
        print("verdict set to '-': a crash produced no evidence")
        verdict = "-"
    s = read_json(SEAL_FILE, {})
    sealed = s.get("exp") == row["exp"]
    pred = ""
    if sealed and status != "crash":
        parts = []
        for metric, lo, hi in predictions(s["text"]):
            v = metrics.get(metric)
            if isinstance(v, (int, float)):
                parts.append(f"{metric}:hit" if lo <= v <= hi else f"{metric}:miss({v:g} vs [{lo:g},{hi:g}])")
        pred = " ".join(parts)
    page_now = read_text(page_path(row["exp"]))
    prereg = "none" if not sealed else ("late" if s["late"] else "blind")
    if sealed and (block(page_now, "Pre-registration") + block(page_now, "Contingencies")).split() != s["text"].split():
        prereg = "edited-after-seal"
    hypos, ideas, preds = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS), read_tsv(PRED_FILE, PRED_COLS)
    tag = row.get("tag", "-")
    hid = "-"
    if tag not in ("", "-"):
        idea = find(ideas, tag, "idea")
        hid = idea["hypothesis"]
        idea["status"] = "queued" if status == "crash" else "done"
        if status == "crash":
            idea["note"] = f"{row['exp']} crashed: fix and relaunch, or drop it"
        write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
        if VERDICTS[verdict]:
            h = find(hypos, hid, "hypothesis")
            h["evidence"] = " ".join(filter(None, [h["evidence"], row["exp"] + VERDICTS[verdict]]))
            write_tsv(HYPO_FILE, HYPO_COLS, hypos)
    preds.append({"exp": row["exp"], "idea": tag or "-", "hypothesis": hid, "status": status,
                  "verdict": verdict, "pred": pred, "prereg": prereg})  # fmt: skip
    write_tsv(PRED_FILE, PRED_COLS, preds)
    print(f"verdict {verdict}; predictions: {pred or '-'}; pre-registration: {prereg}")
    print(f"next: Outcome in notebook/{row['exp']}.md, changelog line + current best in world_model.md, `snapshot`, launch")


def cmd_meta(args):
    metas = read_tsv(META_FILE, META_COLS)
    metas.append({"step": f"M{len(metas) + 1:03d}", "kind": args.kind, "after_exp": last_exp(), "summary": args.summary})
    write_tsv(META_FILE, META_COLS, metas)
    print(f"{args.kind} recorded after {last_exp()}")


def cmd_snapshot(args):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    exp = last_exp()
    if exp != "-" and exp not in read_text(WORLD_MODEL_FILE):
        print(f"warning: world_model.md does not mention {exp} yet", file=sys.stderr)
    dest = snapshot_path(exp if exp != "-" else "E-init")
    with open(dest, "w") as f:
        f.write(read_text(WORLD_MODEL_FILE))
    print(f"saved {os.path.relpath(dest, ROOT)}")


def build_parser():
    p = argparse.ArgumentParser(description="Hypothesis-driven research state (task-agnostic)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    hp = sub.add_parser("hypo").add_subparsers(dest="sub", required=True)
    s = hp.add_parser("add")
    s.add_argument("--statement", required=True)
    s.add_argument("--parent", default="")
    s.add_argument("--confidence", default="0.5")
    s.add_argument("--source", default="")
    s.set_defaults(fn=cmd_hypo_add)
    s = hp.add_parser("set")
    s.add_argument("id")
    s.add_argument("--status", choices=HYPO_STATUSES)
    s.add_argument("--confidence")
    s.add_argument("--statement")
    s.add_argument("--source")
    s.add_argument("--evidence")
    s.set_defaults(fn=cmd_hypo_set)
    hp.add_parser("tree").set_defaults(fn=cmd_hypo_tree)
    ip = sub.add_parser("idea").add_subparsers(dest="sub", required=True)
    s = ip.add_parser("add")
    s.add_argument("--hypothesis", required=True)
    s.add_argument("--description", required=True)
    s.add_argument("--expected", required=True)
    s.add_argument("--priority", type=int, default=3)
    s.set_defaults(fn=cmd_idea_add)
    s = ip.add_parser("next")
    s.add_argument("--peek", action="store_true")
    s.set_defaults(fn=cmd_idea_next)
    s = ip.add_parser("set")
    s.add_argument("id")
    s.add_argument("--status", choices=IDEA_STATUSES)
    s.add_argument("--priority", type=int)
    s.add_argument("--note")
    s.add_argument("--expected")
    s.add_argument("--description")
    s.set_defaults(fn=cmd_idea_set)
    s = ip.add_parser("list")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_idea_list)
    s = sub.add_parser("launch")
    s.add_argument("--idea", required=True)
    s.add_argument("--description", default="")
    s.add_argument("--prereg", default="")
    s.set_defaults(fn=cmd_launch)
    sub.add_parser("prereg").set_defaults(fn=cmd_prereg)
    s = sub.add_parser("log")
    s.add_argument("--status", required=True, choices=["keep", "discard", "crash"])
    s.add_argument("--verdict", required=True, choices=list(VERDICTS))
    s.add_argument("--description", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_log)
    s = sub.add_parser("meta")
    s.add_argument("kind", choices=["ideate", "revise"])
    s.add_argument("--summary", required=True)
    s.set_defaults(fn=cmd_meta)
    sub.add_parser("snapshot").set_defaults(fn=cmd_snapshot)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
