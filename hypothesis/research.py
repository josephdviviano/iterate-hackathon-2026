"""
research.py: the hypothesis-driven research framework's bookkeeping and meta-steps. Task-agnostic.

The experiment agent (one continuous session) uses this file to pop ideas, launch experiments,
pre-register predictions and log verdicts. Experiments are run and judged only by `ar.py`,
exactly as in any other framework.

The two meta-steps are NOT done by the experiment agent. Each is a separate LLM call with exactly
the inputs of the design, started automatically in the background when due:

    ideate:  M(hypotheses, results) -> new ideas
    revise:  M(hypotheses, results, search, review) -> new hypotheses
             (search and review are two more separate calls whose outputs feed revise)

"results" = results.tsv, the experiment agent's verdicts and prediction checks, and the Outcome
and Post-mortem sections of the notebook pages. The task description (task.md) is given to every
call so that its output is about this task.

State (workspace root, untracked by git):
    hypothesis.tsv  hypotheses, Lvl1 (broad claim) > Lvl2 (mechanism) > Lvl3 (falsifiable prediction)
    ideas.tsv       queue of experiment ideas, each testing one hypothesis
    predictions.tsv per experiment: idea, hypothesis, verdict, prediction check, pre-registration
    meta.tsv        journal of the meta-steps, with their cost
    world_model.md  the experiment agent's understanding; read before and updated after every run
    notebook/       E007.md: hypothesis chain, pre-registration, contingencies, outcome, post-mortem
    history/        world model snapshot after every experiment
    drafts/         the next experiment, prepared while the current one runs

    python research.py init                      # state + builds the first hypothesis tree (background)
    python research.py status
    python research.py tree                      # the whole hypothesis tree
    python research.py idea next|list|drop ...
    python research.py launch --idea I004 [--prereg drafts/I004.prereg.md]   (or --idea - --description ...)
    python research.py prereg                    # seal the pre-registration of the running experiment
    python research.py log --status keep|discard|crash --verdict supports|refutes|inconclusive|-
    python research.py meta wait [--max 540]     # wait for running meta-steps
    python research.py snapshot
"""

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
AR = [sys.executable, os.path.join(ROOT, "ar.py")]
TASK_FILE = os.path.join(ROOT, "task.json")
TASK_MD = os.path.join(ROOT, "task.md")
RESULTS_FILE = os.path.join(ROOT, "results.tsv")
STATE_DIR = os.path.join(ROOT, ".ar")
INFLIGHT_FILE = os.path.join(STATE_DIR, "inflight.json")
SEAL_FILE = os.path.join(STATE_DIR, "seal.json")
LOCK_FILE = os.path.join(STATE_DIR, "research.lock")
META_DIR = os.path.join(STATE_DIR, "meta")
JOBS_FILE = os.path.join(STATE_DIR, "meta_jobs.json")
RUNS_DIR = os.path.join(ROOT, "runs")
HYPO_FILE = os.path.join(ROOT, "hypothesis.tsv")
IDEAS_FILE = os.path.join(ROOT, "ideas.tsv")
PRED_FILE = os.path.join(ROOT, "predictions.tsv")
META_FILE = os.path.join(ROOT, "meta.tsv")
WORLD_MODEL_FILE = os.path.join(ROOT, "world_model.md")
NOTEBOOK_DIR = os.path.join(ROOT, "notebook")
HISTORY_DIR = os.path.join(ROOT, "history")
DRAFTS_DIR = os.path.join(ROOT, "drafts")
STATE = ["hypothesis.tsv", "ideas.tsv", "predictions.tsv", "meta.tsv", "world_model.md",
         "notebook/", "history/", "drafts/"]  # fmt: skip

HYPO_COLS = ["id", "level", "parent", "status", "confidence", "evidence", "source", "statement"]
IDEA_COLS = ["id", "hypothesis", "priority", "status", "expected", "description", "note"]
PRED_COLS = ["exp", "idea", "hypothesis", "status", "verdict", "pred", "prereg"]
META_COLS = ["step", "kind", "after_exp", "cost_usd", "summary"]
HYPO_STATUSES = ["open", "supported", "refuted", "inconclusive", "retired"]
CLOSED = ("refuted", "retired")
IDEA_STATUSES = ["queued", "running", "done", "dropped"]
VERDICTS = {"supports": "+", "refutes": "-", "inconclusive": "~", "-": ""}
IDEATE_MIN, IDEATE_REFILL, REVISE_EVERY, PLATEAU = 3, 5, 10, 5

# the meta-step calls: same model and effort as the experiment agent unless overridden
LLM_CMD = os.environ.get("RESEARCH_LLM_CMD", "claude")
META_MODEL = os.environ.get("RESEARCH_META_MODEL", "claude-opus-5-5")
META_EFFORT = os.environ.get("RESEARCH_META_EFFORT", "high")
META_TIMEOUT = int(os.environ.get("RESEARCH_META_TIMEOUT", "2400"))

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

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

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
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(path + ".tmp", path)


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


@contextlib.contextmanager
def locked():
    """Serialize read-modify-write of the research state between the agent and meta-steps."""
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, TypeError):
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


# ---------------------------------------------------------------------------
# hypothesis tree and queue updates (shared by the meta-steps)
# ---------------------------------------------------------------------------


def next_hypo_id(hypos, parent):
    prefix = f"{parent}." if parent else "H"
    taken = [int(h["id"][len(prefix):]) for h in hypos
             if h["id"].startswith(prefix) and h["id"][len(prefix):].isdigit()]  # fmt: skip
    return f"{prefix}{max(taken, default=0) + 1}"


def close_subtree(hypos, ideas, h):
    """A refuted/retired hypothesis retires its open subtree and drops their queued ideas."""
    sub = [d for d in hypos if d["id"].startswith(h["id"] + ".")]
    for d in sub:
        if d["status"] == "open":
            d["status"] = "retired"
    ids = {h["id"]} | {d["id"] for d in sub}
    for i in ideas:
        if i["status"] == "queued" and i["hypothesis"] in ids:
            i["status"], i["note"] = "dropped", f"{h['id']} {h['status']}"


def add_idea(ideas, hypothesis, description, expected, priority):
    nums = [int(i["id"][1:]) for i in ideas if i["id"][1:].isdigit()]
    iid = f"I{max(nums, default=0) + 1:03d}"
    ideas.append({"id": iid, "hypothesis": hypothesis, "priority": str(priority), "status": "queued",
                  "expected": expected, "description": description, "note": ""})  # fmt: skip
    return iid


# ---------------------------------------------------------------------------
# meta-steps: separate LLM calls
# ---------------------------------------------------------------------------

IDEATE_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {"type": "array", "items": {"type": "object", "properties": {
            "hypothesis": {"type": "string"}, "description": {"type": "string"},
            "expected": {"type": "string"}, "priority": {"type": "integer"}},
            "required": ["hypothesis", "description", "expected", "priority"]}},
        "drop": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["id", "reason"]}},
        "summary": {"type": "string"},
    },
    "required": ["ideas", "drop", "summary"],
}  # fmt: skip
SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "claim": {"type": "string"},
            "relevance": {"type": "string"}, "hypotheses": {"type": "array", "items": {"type": "string"}}},
            "required": ["source", "claim", "relevance", "hypotheses"]}},
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
}  # fmt: skip
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "critiques": {"type": "array", "items": {"type": "object", "properties": {
            "target": {"type": "string"}, "issue": {"type": "string"}, "suggestion": {"type": "string"}},
            "required": ["target", "issue", "suggestion"]}},
        "summary": {"type": "string"},
    },
    "required": ["critiques", "summary"],
}  # fmt: skip
REVISE_SCHEMA = {
    "type": "object",
    "properties": {
        "updates": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "status": {"type": "string", "enum": HYPO_STATUSES},
            "confidence": {"type": "number"}, "statement": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["id", "reason"]}},
        "new": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "parent": {"type": "string"}, "statement": {"type": "string"},
            "confidence": {"type": "number"}, "source": {"type": "string"}},
            "required": ["key", "parent", "statement", "confidence", "source"]}},
        "summary": {"type": "string"},
    },
    "required": ["updates", "new", "summary"],
}  # fmt: skip

TREE_RULES = """The hypothesis tree: Lvl1 is a broad claim about what drives the task's objective, Lvl2 a
mechanism under a Lvl1, Lvl3 a falsifiable prediction under a Lvl2 (at most 3 levels). Ids encode
the tree (H2, H2.1, H2.1.3). Evidence tokens: E007+ supports, E007- refutes, E007~ inconclusive."""


def inputs_task():
    t = read_json(TASK_FILE, {})
    obj = t.get("objective", {})
    cons = "; ".join(f"{c['metric']} {c['op']} {c['value']}" for c in t.get("constraints", []))
    return (f"=== TASK (task.md) ===\n{read_text(TASK_MD)}\n=== OBJECTIVE ===\n"
            f"{obj.get('direction')} {obj.get('metric')}; constraints: {cons or 'none'}\n")  # fmt: skip


def inputs_hypotheses():
    return f"=== HYPOTHESES (hypothesis.tsv) ===\n{read_text(HYPO_FILE)}\n"


def inputs_results():
    notes = []
    for r in results():
        text = read_text(page_path(r["exp"]))
        out, pm = section(text, "Outcome"), section(text, "Post-mortem")
        if out or pm:
            notes.append(f"--- {r['exp']} ---\n" + (f"Outcome: {out}\n" if out else "") + (f"Post-mortem: {pm}\n" if pm else ""))
    return (f"=== RESULTS (results.tsv: every experiment, measured by the task) ===\n{read_text(RESULTS_FILE)}\n"
            f"=== VERDICTS AND PREDICTION CHECKS (predictions.tsv) ===\n{read_text(PRED_FILE)}\n"
            f"=== OUTCOMES AND POST-MORTEMS (notebook) ===\n{''.join(notes) or '(none yet)'}\n")  # fmt: skip


def llm(role, prompt, schema, tools=""):
    """One separate, stateless LLM call with structured output. Returns (dict, cost)."""
    os.makedirs(META_DIR, exist_ok=True)
    # the prompt goes on stdin: as an argument it hits the OS limit once the results grow (~128 KB)
    cmd = [LLM_CMD, "-p", "--model", META_MODEL, "--effort", META_EFFORT, "--tools", tools,
           "--no-session-persistence", "--strict-mcp-config", "--output-format", "json",
           "--json-schema", json.dumps(schema)]  # no MCP servers: the prompt is the only input  # fmt: skip
    if tools:
        cmd += ["--allowedTools", tools]
    with tempfile.TemporaryDirectory() as empty:  # nothing on disk to look at: only the prompt
        p = subprocess.run(cmd, cwd=empty, input=prompt, capture_output=True, text=True, timeout=META_TIMEOUT)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    with open(os.path.join(META_DIR, f"{stamp}-{role}.json"), "w") as f:
        f.write(p.stdout or p.stderr)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"{role}: no JSON from the LLM call (exit {p.returncode}): {p.stderr[-300:]}")
    if out.get("is_error") or not isinstance(out.get("structured_output"), dict):
        raise RuntimeError(f"{role}: the call failed: {str(out.get('result'))[:300]}")
    return out["structured_output"], float(out.get("total_cost_usd") or 0.0)


def journal(kind, cost, summary):
    with locked():
        metas = read_tsv(META_FILE, META_COLS)
        metas.append({"step": f"M{len(metas) + 1:03d}", "kind": kind, "after_exp": last_exp(),
                      "cost_usd": f"{cost:.2f}", "summary": summary})  # fmt: skip
        write_tsv(META_FILE, META_COLS, metas)


def run_ideate():
    prompt = f"""ROLE: ideate. You generate experiment ideas for an autonomous research loop.
You get exactly these inputs: the task, the hypothesis tree, the results so far, and the current
idea queue. Nothing else is available to you; do not use tools.

{TREE_RULES}

{inputs_task()}{inputs_hypotheses()}{inputs_results()}=== CURRENT QUEUE (ideas.tsv) ===
{read_text(IDEAS_FILE)}
Write ideas for the open hypotheses (prefer Lvl3 hypotheses without a decisive test yet). Each idea
is ONE concrete change to the files the task lets the experimenter edit, cleanly testing one
hypothesis (give its id). State an expected outcome on the task's metrics, with direction and
rough size. Priority 1-5 (higher runs first) by expected information x expected improvement;
keep roughly 2/3 exploit, 1/3 explore; combine near-misses. Do not repeat ideas that were already
tried or are queued. Drop queued ideas whose hypothesis is settled. Bring the queue to
{IDEATE_REFILL}-8 queued ideas."""
    out, cost = llm("ideate", prompt, IDEATE_SCHEMA)
    added, dropped, skipped = [], [], []
    with locked():
        hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
        ids = {h["id"] for h in hypos}
        for d in out.get("drop", []):
            for i in ideas:
                if i["id"] == d["id"] and i["status"] == "queued":
                    i["status"], i["note"] = "dropped", f"ideate: {d['reason']}"
                    dropped.append(i["id"])
        for idea in out.get("ideas", []):
            h = idea["hypothesis"]
            if h not in ids or closed_in_chain(hypos, h):
                skipped.append(h)
                continue
            added.append(add_idea(ideas, h, idea["description"], idea["expected"],
                                  min(5, max(1, int(idea["priority"])))))  # fmt: skip
        write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
    summary = f"added {', '.join(added) or 'none'}; dropped {', '.join(dropped) or 'none'}" + (
        f"; skipped ideas for unknown/closed {', '.join(skipped)}" if skipped else "") + f". {out.get('summary', '')}"
    journal("ideate", cost, summary)
    return summary


def run_revise():
    tree, res, task = inputs_hypotheses(), inputs_results(), inputs_task()
    search_prompt = f"""ROLE: search. You look outside the research loop for evidence on its hypotheses.
Use web search and fetch (only those tools) to find papers, reports, code or results relevant to
the open hypotheses and to the directions the tree has not considered. Report concrete claims
with their source.

{TREE_RULES}

{task}{tree}{res}"""
    review_prompt = f"""ROLE: review. You are a skeptical reviewer of an autonomous research loop. Do not use tools.
Attack the hypothesis tree against the results: which conclusions rest on a single run or a
difference within noise, which hypotheses the results already contradict, which directions were
never tried, what result was surprising and what would explain it, what the current Lvl1
hypotheses fail to explain.

{TREE_RULES}

{task}{tree}{res}"""
    search, c1 = llm("search", search_prompt, SEARCH_SCHEMA, tools="WebSearch,WebFetch")
    review, c2 = llm("review", review_prompt, REVIEW_SCHEMA)
    revise_prompt = f"""ROLE: revise. You maintain the hypothesis tree of an autonomous research loop.
You get exactly these inputs: the task, the hypotheses, the results, a literature search and a
skeptical review. Nothing else is available to you; do not use tools.

{TREE_RULES}

{task}{tree}{res}=== SEARCH ===
{json.dumps(search, indent=1)}
=== REVIEW ===
{json.dumps(review, indent=1)}

Revise the tree:
1. Roll evidence up: Lvl3 verdicts decide their Lvl2, Lvl2 decide their Lvl1. Update status
   (open / supported / refuted / inconclusive / retired) and confidence honestly, including down.
   A hypothesis should not be closed on a single result within noise.
2. Add new hypotheses where the results, search or review point to mechanisms the tree lacks.
   New items get a "key"; "parent" is an existing id, the key of another new item, or "" for Lvl1.
3. Keep at least 2 Lvl1 hypotheses open. Give every new hypothesis a source
   (results | review | search:<reference>).
If the tree is empty, seed it: 2-4 Lvl1 hypotheses on genuinely different directions, each with
Lvl2 mechanisms and Lvl3 predictions."""
    out, c3 = llm("revise", revise_prompt, REVISE_SCHEMA)
    changes = []
    with locked():
        hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
        by_id = {h["id"]: h for h in hypos}
        for u in out.get("updates", []):
            h = by_id.get(u["id"])
            if not h:
                continue
            if u.get("statement"):
                h["statement"] = u["statement"]
            if u.get("confidence") is not None:
                h["confidence"] = f"{min(1.0, max(0.0, float(u['confidence']))):.2f}"
            if u.get("status") and u["status"] != h["status"]:
                h["status"] = u["status"]
                changes.append(f"{h['id']}->{h['status']}")
                if h["status"] in CLOSED:
                    close_subtree(hypos, ideas, h)
        keys = {}
        for n in out.get("new", []):
            parent = keys.get(n["parent"], n["parent"])
            if parent and (parent not in by_id or parent.count(".") >= 2 or closed_in_chain(hypos, parent)):
                continue
            hid = next_hypo_id(hypos, parent)
            row = {"id": hid, "level": str(hid.count(".") + 1), "parent": parent, "status": "open",
                   "confidence": f"{min(1.0, max(0.0, float(n['confidence']))):.2f}", "evidence": "",
                   "source": n["source"], "statement": n["statement"]}  # fmt: skip
            hypos.append(row)
            by_id[hid] = row
            keys[n["key"]] = hid
            changes.append(f"+{hid}")
        write_tsv(HYPO_FILE, HYPO_COLS, hypos)
        write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
    summary = f"{', '.join(changes) or 'no changes'}. {out.get('summary', '')}"
    journal("revise", c1 + c2 + c3, summary)
    return summary


def jobs():
    return {k: v for k, v in (read_json(JOBS_FILE, {}) or {}).items() if pid_alive(v.get("pid"))}


def start_meta(kind, reason):
    """Start a meta-step in the background (revise is always followed by ideate)."""
    running = jobs()
    if running:
        return None
    os.makedirs(META_DIR, exist_ok=True)
    log = open(os.path.join(META_DIR, f"{time.strftime('%Y%m%dT%H%M%S')}-{kind}.log"), "w")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "meta", "_run", kind], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)  # fmt: skip
    write_json(JOBS_FILE, {kind: {"pid": proc.pid, "started": time.time(), "reason": reason}})
    return kind


def meta_due(hypos, ideas, metas):
    """Which meta-step should run now, if any, and why."""
    reasons = []
    if not [h for h in hypos if h["status"] == "open" and not closed_in_chain(hypos, h["id"])]:
        reasons.append("no open hypotheses")
    revises = [m for m in metas if m["kind"] == "revise"]
    last = exp_num(revises[-1]["after_exp"]) if revises else -1
    if revises and exp_num(last_exp()) - last >= REVISE_EVERY:
        reasons.append(f"{exp_num(last_exp()) - last} experiments since the last revise")
    streak = 0
    for r in reversed(results()):
        if r["status"] == "keep" or exp_num(r["exp"]) <= last:
            break
        streak += 1
    if streak >= PLATEAU:
        reasons.append(f"{streak} experiments without an improvement")
    if reasons:
        return "revise", "; ".join(reasons)
    queued = [i for i in ideas if i["status"] == "queued"]
    if len(queued) < IDEATE_MIN:
        return "ideate", f"{len(queued)} idea(s) queued"
    return None, ""


def auto_meta():
    hypos, ideas, metas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS), read_tsv(META_FILE, META_COLS)
    kind, why = meta_due(hypos, ideas, metas)
    if kind and start_meta(kind, why):
        print(f"meta-step started in the background: {kind} ({why})")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def missing_postmortems(preds):
    out = []
    for p in preds:
        failed = p["status"] in ("discard", "crash") or p["verdict"] == "refutes" or "miss" in p["pred"]
        if p["idea"] != "-" and failed and not section(read_text(page_path(p["exp"])), "Post-mortem"):
            out.append(p["exp"])
    return out


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
    if not args.no_seed:
        start_meta("revise", "seed the hypothesis tree")
        print("seeding the hypothesis tree in the background (revise, then ideate); run the baseline meanwhile")


def cmd_status(args):
    hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
    preds, metas = read_tsv(PRED_FILE, PRED_COLS), read_tsv(META_FILE, META_COLS)
    ar("status")
    blind = [p for p in preds if p["pred"] and p["prereg"] == "blind"][-10:]
    for metric in metric_names():
        marks = re.findall(rf"\b{re.escape(metric)}:(hit|miss)", " ".join(p["pred"] for p in blind))
        if marks:
            print(f"prediction calibration, {metric}: {marks.count('hit')}/{len(marks)} hit (last 10 blind)")
    print("\nopen hypotheses (`tree` shows all):")
    print_tree(hypos, only_open=True)
    queue = sorted([i for i in ideas if i["status"] in ("queued", "running")],
                   key=lambda i: (i["status"] != "running", -int(i["priority"] or 0), i["id"]))  # fmt: skip
    print(f"\nqueue ({sum(i['status'] == 'queued' for i in ideas)} queued):")
    for i in queue[:8]:
        print(f"  {i['id']} p{i['priority']} {i['status']:<8} {i['hypothesis']:<8} {i['description']}  -> {i['expected']}")
    running = jobs()
    for kind, j in running.items():
        print(f"\nmeta-step running: {kind} ({j['reason']}), {time.time() - j['started']:.0f}s")
    if metas:
        m = metas[-1]
        print(f"last meta-step: {m['step']} {m['kind']} after {m['after_exp']} (${m['cost_usd']}): {m['summary'][:200]}")
        cost = sum(float(x["cost_usd"] or 0) for x in metas)
        print(f"meta-steps so far: {len(metas)}, ${cost:.2f}")
    fl = read_json(INFLIGHT_FILE)
    pm = missing_postmortems(preds)
    if fl:
        seal = read_json(SEAL_FILE, {})
        print(f"\nwhile {fl['exp']} runs, in this order:")
        steps = [] if seal.get("exp") == fl["exp"] else [f"pre-register {fl['exp']} in notebook/{fl['exp']}.md, then `prereg`"]
        if results():
            steps.append(f"world model: finish the full update for {last_exp()}, then `snapshot` again")
        if pm:
            steps.append(f"post-mortems: {', '.join(pm)}")
        steps.append("draft the next idea (patch + pre-registration) in drafts/, then `python ar.py wait`")
        for n, s in enumerate(steps, 1):
            print(f"  {n}. {s}")
    else:
        if results() and not os.path.exists(snapshot_path(last_exp())):
            print(f"\nbefore the next launch: update world_model.md for {last_exp()}, then `snapshot`")
        if not [i for i in ideas if i["status"] in ("queued", "running")]:
            print("the queue is empty: " + ("wait for the meta-step (`meta wait`)" if running else "run `meta wait` (one will start)"))


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


def cmd_tree(args):
    print_tree(read_tsv(HYPO_FILE, HYPO_COLS))


def cmd_idea_next(args):
    with locked():
        hypos, ideas = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS)
        running = [i for i in ideas if i["status"] == "running"]
        if running:
            i = running[0]
        else:
            queued = [i for i in ideas if i["status"] == "queued" and not closed_in_chain(hypos, i["hypothesis"])]
            if not queued:
                auto_meta()
                die("the queue is empty; a meta-step is filling it: `python research.py meta wait`")
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


def cmd_idea_drop(args):
    with locked():
        ideas = read_tsv(IDEAS_FILE, IDEA_COLS)
        i = find(ideas, args.id, "idea")
        if i["status"] not in ("queued", "running"):
            die(f"{i['id']} is {i['status']}")
        i["status"], i["note"] = "dropped", args.reason
        write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
    print(f"{i['id']} dropped")


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
    if results() and not os.path.exists(snapshot_path(last_exp())):
        die(f"update world_model.md for {last_exp()}, then `python research.py snapshot`")
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
    if args.prereg and not (block(draft, "Pre-registration") and block(draft, "Contingencies")):
        die(f"{args.prereg} needs the '## Pre-registration' and '## Contingencies' sections, "
            "formatted as in a notebook page")  # fmt: skip
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
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                parts.append(f"{metric}:hit" if lo <= v <= hi else f"{metric}:miss({v:g} vs [{lo:g},{hi:g}])")
        pred = " ".join(parts)
    page_now = read_text(page_path(row["exp"]))
    prereg = "none" if not sealed else ("late" if s["late"] else "blind")
    if sealed and (block(page_now, "Pre-registration") + block(page_now, "Contingencies")).split() != s["text"].split():
        prereg = "edited-after-seal"
    tag, hid = row.get("tag", "-"), "-"
    with locked():
        hypos, ideas, preds = read_tsv(HYPO_FILE, HYPO_COLS), read_tsv(IDEAS_FILE, IDEA_COLS), read_tsv(PRED_FILE, PRED_COLS)
        if tag not in ("", "-"):
            idea = find(ideas, tag, "idea")
            hid = idea["hypothesis"]
            idea["status"] = "queued" if status == "crash" else "done"
            if status == "crash":
                idea["note"] = f"{row['exp']} crashed: fix and relaunch, or drop it"
            write_tsv(IDEAS_FILE, IDEA_COLS, ideas)
            if VERDICTS[verdict]:
                for h in hypos:
                    if h["id"] == hid:
                        h["evidence"] = " ".join(filter(None, [h["evidence"], row["exp"] + VERDICTS[verdict]]))
                write_tsv(HYPO_FILE, HYPO_COLS, hypos)
        preds.append({"exp": row["exp"], "idea": tag or "-", "hypothesis": hid, "status": status,
                      "verdict": verdict, "pred": pred, "prereg": prereg})  # fmt: skip
        write_tsv(PRED_FILE, PRED_COLS, preds)
    print(f"verdict {verdict}; predictions: {pred or '-'}; pre-registration: {prereg}")
    print(f"next: Outcome in notebook/{row['exp']}.md, changelog line + current best in world_model.md, `snapshot`, launch")
    auto_meta()


def cmd_meta(args):
    if args.action == "_run":  # the background process
        kind = args.kind
        try:
            print(run_revise() if kind == "revise" else run_ideate(), flush=True)
            if kind == "revise":
                kind = "ideate"
                print(run_ideate(), flush=True)
        except Exception as exc:  # recorded, so `status` shows it; the next log retries
            journal(f"{kind}-failed", 0.0, f"{type(exc).__name__}: {exc}"[:400])
            raise
        finally:
            write_json(JOBS_FILE, {})
        return
    if args.action == "wait":
        if not jobs():
            auto_meta()
        end = time.time() + args.max
        while jobs() and time.time() < end:
            time.sleep(5)
        if jobs():
            print(f"still running after {args.max:.0f}s: {', '.join(jobs())}; call `meta wait` again")
            sys.exit(3)
        metas = read_tsv(META_FILE, META_COLS)
        print(f"no meta-step running; last: {metas[-1]['kind']}: {metas[-1]['summary'][:300]}" if metas else "no meta-step ran")


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
    p = argparse.ArgumentParser(description="Hypothesis-driven research framework (task-agnostic)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-seed", action="store_true", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_init)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("tree").set_defaults(fn=cmd_tree)
    ip = sub.add_parser("idea").add_subparsers(dest="sub", required=True)
    s = ip.add_parser("next")
    s.add_argument("--peek", action="store_true")
    s.set_defaults(fn=cmd_idea_next)
    s = ip.add_parser("drop")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    s.set_defaults(fn=cmd_idea_drop)
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
    s.add_argument("action", choices=["wait", "_run"])
    s.add_argument("kind", nargs="?", choices=["ideate", "revise"])
    s.add_argument("--max", type=float, default=540)
    s.set_defaults(fn=cmd_meta)
    sub.add_parser("snapshot").set_defaults(fn=cmd_snapshot)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
