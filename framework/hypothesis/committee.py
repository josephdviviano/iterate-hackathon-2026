"""
committee.py: a committee of verified world models for the hypothesis framework. Task-agnostic.

After the program committee of iterate-hackathon-2026 (branch jdv, src/committee): uncertainty is
the disagreement among independently written world models that are each consistent with every
observation so far. Unverified LLM opinions do not carry that signal; verified programs do.

    configuration  each experiment's code is described as flat settings (one small LLM call per
                   experiment, from the diff against the code it was compared with); a queued idea
                   is described as the settings it would produce on top of the current best
    member         a Python program predict(config) -> metrics, written by one LLM call from one
                   premise assumption; members never see each other
    admission      a member is admitted only if it replays the results: for each past experiment it
                   predicts the outcome class (improved / not improved / missed a constraint, from its
                   own predictions for the run and for the code it was compared with) and the
                   objective within REPLAY_TOL. Measurements are noisy, so a member must replay ADMIT
                   of the experiments rather than all of them. A rejected candidate is shown its
                   mismatches and repaired (counterexample-guided), up to REPAIRS times
    vote           every admitted member predicts each queued idea; disagreement is the normalised
                   entropy of their outcome votes (0 = unanimous, 1 = evenly split)
    actions        the most disputed idea is raised (running it decides between world models), while
                   disagreement is measured to predict the vote's errors; a new result that no member
                   replays is a counterexample: it makes a revise due, steers the next literature
                   question, and half the committee is rewritten around it; members that stop
                   replaying the data are retired and replaced; a committee whose members all vote
                   alike is renewed; an idea every member says misses a constraint is dropped (every
                   VETO_AUDIT_EVERY-th such veto is run anyway as an audit)
    calibration    each vote is scored on its result: AUROC of disagreement against vote errors,
                   error when unanimous vs split, and adaptive conformal coverage (Gibbs and Candes)

State (workspace root, untracked): committee/{schema.json, configs.jsonl, members.json, members/W1.py,
forecasts.jsonl, ledger.jsonl, rounds.jsonl, state.json, report.md}
"""

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research as R  # noqa: E402

DIR = os.path.join(R.ROOT, "committee")
MEMBER_DIR = os.path.join(DIR, "members")
SCHEMA_FILE = os.path.join(DIR, "schema.json")
CONFIGS_FILE = os.path.join(DIR, "configs.jsonl")
MEMBERS_FILE = os.path.join(DIR, "members.json")
FORECASTS_FILE = os.path.join(DIR, "forecasts.jsonl")
LEDGER_FILE = os.path.join(DIR, "ledger.jsonl")
ROUNDS_FILE = os.path.join(DIR, "rounds.jsonl")
STATE_FILE = os.path.join(DIR, "state.json")
REPORT_FILE = os.path.join(DIR, "report.md")

MODEL = os.environ.get("RESEARCH_COMMITTEE_MODEL", "claude-sonnet-5-5")
EFFORT = os.environ.get("RESEARCH_COMMITTEE_EFFORT", "medium")
ADMIT = 0.85  # share of past experiments a member must replay
REPLAY_TOL = 0.05  # relative tolerance on the objective when replaying
REPAIRS = 2  # counterexample-guided repairs of a rejected candidate
MIN_TRANSITIONS = 5  # experiments needed before members are written
MAX_SOURCE = 8000  # characters: a world model, not a lookup table of the results
VETO_AUDIT_EVERY = 4
COLLAPSE_ROUNDS = 3
AUROC_MIN, AUROC_N = 0.55, 12
REVISE_GAP = 3  # experiments between counterexample-driven revises
NOISE_FLOOR = 0.01  # relative noise band when there are no reruns to estimate it from
MAX_EXACT_TESTS = 4  # tests of a setting against an exact number: more is a lookup table of experiments
MAX_ROWS = 60  # experiments shown to a member being written (it is checked on all of them)
CEGIS_N = 3  # candidates written around a counterexample
CLASSES = ["improved", "not-improved", "infeasible"]

GENERIC_PREMISES = [
    "Account for the objective as the sum of the costs of the process's parts. Derive each part's cost from "
    "the settings that scale it (sizes, counts, steps), and the constrained metrics from the budget those parts buy.",
    "Prefer the simplest model that fits: each setting moves each metric by a fixed amount or factor, "
    "fitted from the experiments where only that setting changed.",
    "Assume interactions dominate: the effect of one setting depends on others. Find the pairs that "
    "matter in the data and model them explicitly.",
    "Assume diminishing returns and thresholds: metrics saturate with budget and fall off a cliff past "
    "some setting. Locate the knees from the data.",
    "Start from the experiments that missed a constraint or failed to improve. Write the boundary that "
    "separates them from the rest first, then the objective inside it.",
    "Assume the measured differences between close configurations are mostly noise, and that only a few "
    "settings truly matter. Find those few and ignore the rest.",
]

RUNNER = r"""
import sys, json, importlib.util, traceback, math
spec = importlib.util.spec_from_file_location("member", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    json.dump({"load_error": traceback.format_exc()[-600:]}, open(sys.argv[3], "w")); sys.exit(0)
out = {}
for k, cfg in json.load(open(sys.argv[2])).items():
    try:
        p = mod.predict(dict(cfg))
        out[k] = {m: float(v) for m, v in p.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
    except Exception:
        out[k] = {"__error__": traceback.format_exc()[-300:]}
json.dump(out, open(sys.argv[3], "w"))
"""
FORBIDDEN = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "input", "breakpoint", "vars"}

KV = {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": ["number", "string", "boolean"]}},
      "required": ["key", "value"]}  # fmt: skip
NEWKEY = {"type": "object", "properties": {"key": {"type": "string"}, "description": {"type": "string"}},
          "required": ["key", "description"]}  # fmt: skip
RUN_SCHEMA = {"type": "object", "properties": {"config": {"type": "array", "items": KV},
              "new_keys": {"type": "array", "items": NEWKEY}}, "required": ["config", "new_keys"]}  # fmt: skip
IDEAS_SCHEMA = {"type": "object", "properties": {
    "ideas": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"},
              "config": {"type": "array", "items": KV}}, "required": ["id", "config"]}},
    "new_keys": {"type": "array", "items": NEWKEY}}, "required": ["ideas", "new_keys"]}  # fmt: skip
SYNTH_SCHEMA = {"type": "object", "properties": {"source": {"type": "string"}, "mechanisms": {"type": "string"}},
                "required": ["source", "mechanisms"]}  # fmt: skip


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def config():
    return (R.read_json(R.FRAMEWORK_FILE, {}) or {}).get("committee") or {}


def enabled():
    return bool(config())


def K():
    return int(config().get("k", 6))


def read_jsonl(path):
    out = []
    for line in R.read_text(path).splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_jsonl(path, rows):
    os.makedirs(DIR, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def state():
    return R.read_json(STATE_FILE, {}) or {}


def set_state(**kw):
    R.write_json(STATE_FILE, {**state(), **kw})


def load_ar():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ar_for_committee", os.path.join(R.ROOT, "ar.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def task():
    return R.read_json(R.TASK_FILE)


def numeric_metrics(t):
    out = [t["objective"]["metric"]]
    if t.get("infeasible_order"):
        out.append(t["infeasible_order"]["metric"])
    out += [c["metric"] for c in t.get("constraints", []) if not isinstance(c["value"], bool)]
    return list(dict.fromkeys(out))


def members(active_only=True):
    ms = R.read_json(MEMBERS_FILE, []) or []
    return [m for m in ms if m["status"] == "active"] if active_only else ms


def source_of(mid):
    return R.read_text(os.path.join(MEMBER_DIR, f"{mid}.py"))


def git(*a):
    return subprocess.run(["git", *a], cwd=R.ROOT, capture_output=True, text=True).stdout


def outcome_class(row):
    return "improved" if row["improved"] == "yes" else ("infeasible" if row["feasible"] != "yes" else "not-improved")


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------


def configs():
    """{key: config} where key is an experiment (E012) or an idea at a tip (I034@E030); newest wins."""
    out = {}
    for row in read_jsonl(CONFIGS_FILE):
        out[row["key"]] = row["config"]
    return out


def schema():
    return R.read_json(SCHEMA_FILE, {}) or {}


def add_keys(new_keys):
    s = schema()
    for k in new_keys:
        s.setdefault(k["key"], k["description"][:200])
    R.write_json(SCHEMA_FILE, s)


def as_dict(kvs):
    return {kv["key"]: kv["value"] for kv in kvs if kv.get("key") and not kv["key"].startswith("__")}


def plain(cfg):
    return {k: v for k, v in (cfg or {}).items() if not k.startswith("__")}


def editable_files(ref):
    out = []
    for path in git("ls-tree", "-r", "--name-only", ref, "--", *task().get("editable", [])).split():
        body = git("show", f"{ref}:{path}")
        if "\0" not in body:
            out.append(f"--- {path} ---\n{body[:40000]}")
    return "\n".join(out)[:60000]


def featurize_prompt(what):
    keys = "\n".join(f"{k}: {v}" for k, v in schema().items()) or "(none yet: define them)"
    return f"""ROLE: committee-featurize. You describe experiment configurations as flat settings for predictive models.
Do not use tools.

{R.inputs_task()}
=== KNOWN SETTINGS (reuse these keys; add a key only for a setting none of them covers) ===
{keys}

{what}

Rules: numbers as numbers, choices as short strings, switches as booleans; a new key gets a one-line
description. Capture the settings that plausibly drive the task's metrics: sizes, counts, schedules,
rates, precision, algorithm choices. Never encode an experiment's result or name as a setting.
{"Give the COMPLETE configuration (10-40 keys)." if "=== THE CODE" in what else
 "Give ONLY the settings the change alters: their new values, and new keys for settings the change introduces. Do not restate unchanged settings or re-describe the parent with other keys."}"""


def featurize_run(exp, ref, parent_key, parent_cfg):
    if parent_cfg is None:
        what = f"=== THE CODE ({exp}) ===\n{editable_files(ref)}\n\nReturn the configuration of this code."
    else:
        diff = git("diff", "-U2", parent_cfg["__commit__"], ref, "--", *task().get("editable", []))[:30000]
        what = (f"=== PARENT CONFIGURATION ({parent_key}: the code this change was applied to) ===\n"
                f"{json.dumps(plain(parent_cfg))}\n=== THE CHANGE ({exp}, as a diff of the code) ===\n"
                f"{diff or '(no change)'}\n\nReturn the settings this change alters.")  # fmt: skip
    out, cost = R.llm(f"committee-featurize-{exp}", featurize_prompt(what), RUN_SCHEMA, model=MODEL, effort=EFFORT)
    add_keys(out.get("new_keys", []))
    return {**plain(parent_cfg), **as_dict(out.get("config", [])), "__commit__": ref}, cost  # the parent plus the delta


def featurize_ideas(ideas, tip_key, tip_cfg):
    listing = "\n".join(f"{i['id']}: {i['description']}" for i in ideas)
    what = (f"=== PARENT CONFIGURATION ({tip_key}: the current best) ===\n{json.dumps(plain(tip_cfg))}\n"
            f"=== PROPOSED CHANGES (not run yet) ===\n{listing}\n\n"
            "For each proposed change, return the settings it would alter if implemented as described on top of "
            "the parent (best guess for details it leaves open). Use the ids exactly as given.")  # fmt: skip
    out, cost = R.llm("committee-featurize-ideas", featurize_prompt(what), IDEAS_SCHEMA, model=MODEL, effort=EFFORT)
    add_keys(out.get("new_keys", []))
    res = {}
    for x in out.get("ideas", []):
        m = re.match(r"\s*(I\d+)", x["id"])  # some models append the description to the id
        if m:
            res[m.group(1)] = {**plain(tip_cfg), **as_dict(x["config"]), "__commit__": tip_cfg.get("__commit__")}
    return res, cost


def tip_before(rows, idx):
    for r in reversed(rows[:idx]):
        if r["status"] == "keep":
            return r
    return None


def featurize_new_runs():
    """Configurations for logged experiments that lack one, in order (a parent is described first)."""
    rows, cfg, cost, done = R.results(), configs(), 0.0, []
    for idx, r in enumerate(rows):
        if r["exp"] in cfg or r["status"] == "crash":
            continue
        parent = tip_before(rows, idx)
        if parent and parent["exp"] not in cfg:
            continue  # its parent could not be described: retried next round
        pcfg = cfg.get(parent["exp"]) if parent else None
        if pcfg and pcfg.get("__commit__") == r["commit"]:
            c = dict(pcfg)  # a rerun of the same code
        else:
            try:
                c, k = featurize_run(r["exp"], r["commit"], parent["exp"] if parent else None, pcfg)
                cost += k
            except Exception as exc:
                R.journal("committee-failed", 0.0, f"featurize {r['exp']}: {type(exc).__name__}: {exc}"[:300])
                continue
        append_jsonl(CONFIGS_FILE, [{"key": r["exp"], "config": c, "ts": time.time()}])
        cfg[r["exp"]] = c
        done.append(r["exp"])
    return done, cost


def current_tip():
    t = R.tip()
    return (t["exp"], configs().get(t["exp"])) if t else (None, None)


def featurize_queue(ideas, tip_key, tip_cfg):
    if not tip_cfg:
        return {}, 0.0
    cfg, cost = configs(), 0.0
    need = [i for i in ideas if f"{i['id']}@{tip_key}" not in cfg]
    if need:
        try:
            res, cost = featurize_ideas(need, tip_key, tip_cfg)
            append_jsonl(CONFIGS_FILE, [{"key": f"{iid}@{tip_key}", "config": c, "ts": time.time()} for iid, c in res.items()])
            cfg = configs()
        except Exception as exc:
            R.journal("committee-failed", 0.0, f"featurize ideas: {type(exc).__name__}: {exc}"[:300])
    return {i["id"]: cfg[f"{i['id']}@{tip_key}"] for i in ideas if f"{i['id']}@{tip_key}" in cfg}, cost


# ---------------------------------------------------------------------------
# transitions, execution and verification
# ---------------------------------------------------------------------------


def transitions():
    """Each logged experiment with a configuration whose parent (the code it was compared with) has one."""
    ar, t = load_ar(), task()
    rows, cfg, out = R.results(), configs(), []
    for idx, r in enumerate(rows):
        parent = tip_before(rows, idx)
        if r["status"] == "crash" or not parent or r["exp"] not in cfg or parent["exp"] not in cfg:
            continue
        m = ar.row_metrics(t, r)
        if ar.valid(t, m):
            out.append({"exp": r["exp"], "parent": parent["exp"], "measured": m, "parent_measured": ar.row_metrics(t, parent),
                        "class": outcome_class(r)})  # fmt: skip
    return out


def noise_band():
    """Relative run-to-run noise per metric, from reruns of unchanged code (twice the median difference);
    NOISE_FLOOR without at least two reruns. Outcome classes inside this band are a coin flip."""
    ar, t = load_ar(), task()
    rows, diffs = R.results(), {m: [] for m in numeric_metrics(t)}
    for idx, r in enumerate(rows):
        p = tip_before(rows, idx)
        if p and r["commit"] == p["commit"] and r["status"] != "crash":
            a, b = ar.row_metrics(t, r), ar.row_metrics(t, p)
            for m in diffs:
                if isinstance(a.get(m), (int, float)) and isinstance(b.get(m), (int, float)) and b[m]:
                    diffs[m].append(abs(a[m] - b[m]) / abs(b[m]))
    out = {}
    for m, v in diffs.items():
        v = sorted(v)
        out[m] = max(NOISE_FLOOR, 2 * v[len(v) // 2]) if len(v) >= 2 else NOISE_FLOOR
    return out


def accepted_classes(t, tr, band):
    """The outcome classes consistent with a measurement, given the noise: a near-tie with the parent may be
    either improved or not; a run near a constraint's line may be either side of it."""
    ok = {tr["class"]}
    m, p = tr["measured"], tr.get("parent_measured") or {}
    obj = t["objective"]
    order = t.get("infeasible_order") or obj
    for spec in (obj, order):
        a, b = m.get(spec["metric"]), p.get(spec["metric"])
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b and abs(a - b) / abs(b) <= band.get(spec["metric"], NOISE_FLOOR):
            ok |= {"improved", "not-improved"} if tr["class"] != "infeasible" else set()
    for c in t.get("constraints", []):
        v = m.get(c["metric"])
        if isinstance(v, (int, float)) and not isinstance(c["value"], bool) and c["value"] and \
                abs(v - c["value"]) / abs(c["value"]) <= band.get(c["metric"], NOISE_FLOOR):
            ok |= {"infeasible", "improved", "not-improved"}  # feasibility itself is within noise
    return ok


def static_problems(source):
    if not isinstance(source, str):
        return "no source"
    if len(source) > MAX_SOURCE:
        return f"source is {len(source)} characters (limit {MAX_SOURCE}): model mechanisms, not a table of results"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    exact = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(n != "math" for n in names):
                return f"only `import math` is allowed (found {names})"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN:
            return f"`{node.id}` is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attributes are not allowed"
        if isinstance(node, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for o in node.ops):
            for side in [node.left, *node.comparators]:
                elts = side.elts if isinstance(side, (ast.Tuple, ast.List, ast.Set)) else [side]
                exact += sum(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) and not isinstance(e.value, bool)
                             for e in elts)  # fmt: skip
    if exact > MAX_EXACT_TESTS:
        return (f"{exact} tests of settings against exact numbers (limit {MAX_EXACT_TESTS}): that looks up experiments; "
                "model how the settings drive the metrics instead")  # fmt: skip
    if not any(isinstance(n, ast.FunctionDef) and n.name == "predict" for n in tree.body):
        return "no top-level `def predict(config)`"
    return None


def execute(source, cfgs, timeout=30):
    """predict() on every configuration, in an isolated process. ({key: {metric: value}}, error)."""
    bad = static_problems(source)
    if bad:
        return None, bad
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in (("member.py", source), ("run.py", RUNNER), ("cfgs.json", json.dumps({k: plain(c) for k, c in cfgs.items()}))):
            with open(os.path.join(tmp, name), "w") as f:
                f.write(body)
        try:
            subprocess.run([sys.executable, "-I", "run.py", "member.py", "cfgs.json", "out.json"], cwd=tmp,
                           capture_output=True, text=True, timeout=timeout)  # fmt: skip
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout}s"
        out = R.read_json(os.path.join(tmp, "out.json"))
    if out is None:
        return None, "the program produced no output"
    if "load_error" in out:
        return None, "load error: " + out["load_error"]
    return out, None


def classify(ar, t, pred, parent_pred):
    """Outcome class of a predicted run, judged against the predicted code it is compared with."""
    def full(p):
        m = dict(p)
        for c in t.get("constraints", []):
            if isinstance(c["value"], bool):
                m[c["metric"]] = c["value"]  # switches (e.g. completion) are assumed to hold
        return m

    cand, base = full(pred), full(parent_pred)
    if not ar.valid(t, cand):
        return None
    if ar.improves(t, cand, base if ar.valid(t, base) else None)[0]:
        return "improved"
    return "not-improved" if ar.feasible(t, cand) else "infeasible"


def replay(source, trans):
    """Verify a member on every transition: (pass list, mismatch lines, error). A class within measurement
    noise of the parent or of a constraint counts as replayed (accepted_classes)."""
    ar, t, cfg, band = load_ar(), task(), configs(), noise_band()
    keys = {tr["exp"] for tr in trans} | {tr["parent"] for tr in trans}
    out, err = execute(source, {k: cfg[k] for k in keys if k in cfg})
    if err:
        return [], [err], err
    obj, mets = t["objective"]["metric"], numeric_metrics(t)
    passes, mism = [], []
    for tr in trans:
        p, q = out.get(tr["exp"], {}), out.get(tr["parent"], {})
        if "__error__" in p or "__error__" in q:
            passes.append(False)
            mism.append(f"{tr['exp']}: predict() raised {(p.get('__error__') or q.get('__error__'))[-160:]}")
            continue
        cls = classify(ar, t, p, q)
        v, pv = tr["measured"].get(obj), p.get(obj)
        num_ok = isinstance(pv, float) and isinstance(v, (int, float)) and abs(pv - v) <= REPLAY_TOL * abs(v)
        ok = cls in accepted_classes(t, tr, band) and num_ok
        passes.append(ok)
        if not ok:
            got = ", ".join(f"{m} {p.get(m, float('nan')):.4g}" for m in mets)
            real = ", ".join(f"{m} {tr['measured'].get(m)}" for m in mets)
            mism.append(f"{tr['exp']} (vs {tr['parent']}): predicted {cls} ({got}); measured {tr['class']} ({real})")
    return passes, mism, None


# ---------------------------------------------------------------------------
# synthesis: one LLM call per candidate, counterexample-guided repairs
# ---------------------------------------------------------------------------


def transitions_table(trans):
    """The experiments a member is written from: the most recent MAX_ROWS plus every open counterexample, with
    only the settings that vary across them as columns (the constant ones are listed once)."""
    cfg, t = configs(), task()
    keep = set(state().get("open_counterexamples", []))
    rows = [tr for i, tr in enumerate(trans) if i >= len(trans) - MAX_ROWS or tr["exp"] in keep]
    values = {}
    for tr in rows:
        for k, v in plain(cfg.get(tr["exp"])).items():
            values.setdefault(k, set()).add(json.dumps(v))
    vary = sorted(k for k, v in values.items() if len(v) > 1)
    const = {k: json.loads(next(iter(v))) for k, v in values.items() if len(v) == 1}
    mets = numeric_metrics(t)
    lines = ["exp\tcompared_with\toutcome\t" + "\t".join(mets) + "\t" + "\t".join(vary)]
    for tr in rows:
        c = cfg[tr["exp"]]
        lines.append(f"{tr['exp']}\t{tr['parent']}\t{tr['class']}\t" + "\t".join(str(tr["measured"].get(m)) for m in mets)
                     + "\t" + "\t".join(str(c.get(k, "")) for k in vary))  # fmt: skip
    note = f"(Showing {len(rows)} of {len(trans)} experiments; your program is checked on all of them.)" if len(rows) < len(trans) else ""
    return "\n".join(lines) + ("\n" + note if note else ""), json.dumps(const)


def synth_prompt(premise, trans, prior=None, mismatches=None, counterexample=None):
    t = task()
    table, const = transitions_table(trans)
    sch = schema()
    used = set(json.loads(const)) | set(table.split("\n", 1)[0].split("\t")[3:])
    keys = "\n".join(f"{k}: {v}" for k, v in sch.items() if k in used)
    mets = numeric_metrics(t)
    repair = ""
    if prior:
        repair = (f"\n=== YOUR PREVIOUS PROGRAM ===\n{prior}\n=== EXPERIMENTS IT DOES NOT REPLAY ===\n"
                  + "\n".join((mismatches or [])[:15]) + "\nRevise the program so that it replays these too, without "
                  "breaking the others. Change the mechanism, not just constants for single experiments.\n")  # fmt: skip
    cex = (f"\n=== COUNTEREXAMPLE (no current world model replays this; your program MUST replay it) ===\n{counterexample}\n"
           if counterexample else "")  # fmt: skip
    return f"""ROLE: committee-synth. You write a world model of an autonomous research task as an executable
Python program. Do not use tools.

{R.inputs_task()}
=== SETTINGS (configuration keys) ===
{keys}
=== SETTINGS THAT ARE THE SAME IN EVERY EXPERIMENT BELOW ===
{const}
=== EXPERIMENTS (each run, the code it was compared with, its outcome, metrics and the settings that vary) ===
{table}
{cex}
=== YOUR STARTING ASSUMPTION ===
{premise}
{repair}
Write a Python module with `def predict(config: dict) -> dict` that returns a float for each of
{', '.join(mets)}, for ANY configuration (use sensible defaults for missing keys). Only `import math` is
allowed; at most {MAX_SOURCE} characters; no file, network or system access; at most {MAX_EXACT_TESTS} tests of a
setting against an exact number. Model the mechanisms by which the settings drive the metrics, so it
generalises to new configurations; do not look up experiments.
It will be checked by replaying every experiment: from your predictions for the run and for the code it was
compared with, the outcome class (improved / not-improved / infeasible, judged by the task's objective and
constraints; differences within measurement noise may go either way) must match, and the predicted
{t['objective']['metric']} must be within {REPLAY_TOL:.0%} of the measured value, for at least {ADMIT:.0%} of the experiments.
Return the source, and `mechanisms`: at most 80 words on what you believe drives the metrics."""


def synthesize(mid, premise, trans, counterexample=None, must=None, warm=None):
    """One candidate: write (or start from a warm source), verify, repair up to REPAIRS times.
    `must`: an experiment the candidate has to replay (the counterexample). (member or None, cost, log, best source)."""
    cost, prior, mism, log, best = 0.0, None, None, [], (0.0, None)

    def check(src):
        passes, mm, err = replay(src, trans)
        rate = sum(passes) / len(passes) if passes else 0.0
        ok_must = must is None or any(p and tr["exp"] == must for p, tr in zip(passes, trans))
        return passes, mm, err, rate, ok_must

    def admit(src, mech, rate, attempts):
        os.makedirs(MEMBER_DIR, exist_ok=True)
        with open(os.path.join(MEMBER_DIR, f"{mid}.py"), "w") as f:
            f.write(src)
        return {"id": mid, "premise": premise, "mechanisms": str(mech or "")[:600], "status": "active",
                "born": R.last_exp(), "replay": round(rate, 3), "attempts": attempts}  # fmt: skip

    if warm:  # the best earlier attempt from this premise: re-check it on the new data first, without a call
        _, mm, err, rate, ok_must = check(warm["source"])
        if not err and rate >= ADMIT and ok_must:
            return admit(warm["source"], warm.get("mechanisms"), rate, 0), 0.0, ["admitted the earlier attempt on new data"], None
        prior, mism, best = warm["source"], mm, (rate, warm)
    for attempt in range(REPAIRS + 1):
        try:
            out, c = R.llm(f"committee-synth-{mid}", synth_prompt(premise, trans, prior, mism, counterexample),
                           SYNTH_SCHEMA, model=MODEL, effort=EFFORT)  # fmt: skip
            cost += c
            src = out.get("source")
            _, mism, err, rate, ok_must = check(src)
        except Exception as exc:
            log.append(f"attempt {attempt + 1} failed: {exc}"[:200])
            break
        log.append(f"attempt {attempt + 1}: " + (err[:160] if err else f"replays {rate:.0%}"
                                                  + ("" if ok_must else f", not the counterexample {must}")))  # fmt: skip
        if not err and rate >= ADMIT and ok_must:
            return admit(src, out.get("mechanisms"), rate, attempt + 1), cost, log, None
        if not err and rate >= best[0]:
            best = (rate, {"source": src, "mechanisms": str(out.get("mechanisms") or "")[:600], "rate": round(rate, 3)})
        prior = src
    return None, cost, log, best[1]


def premises_for(n, used):
    """Starting assumptions: the generic ones first (data-driven, so new members are not seeded with what current
    members already believe), then open hypotheses nearest 0.5 confidence; one slot goes to the newest literature
    digest when there is one."""
    hypos = [h for h in R.read_tsv(R.HYPO_FILE, R.HYPO_COLS) if h["status"] == "open" and h["level"] in ("1", "2")]
    hypos.sort(key=lambda h: abs(float(h["confidence"] or 0.5) - 0.5))
    pool = GENERIC_PREMISES + [f"Assume this hypothesis is the main mechanism: {h['statement']}" for h in hypos[:3]]
    fresh = [s for s in pool if s not in used] or pool
    out = fresh[:n]
    lit = literature_premise(used)
    if lit and out:
        out[-1] = lit
    return out


def literature_premise(used):
    if R.lit_mode() == "off":
        return None
    ddir = os.path.join(R.LIT_DIR, "digests")
    names = sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []
    if not names:
        return None
    p = (f"Assume what this literature digest ({names[-1][:-3]}) reports holds in this task, and build on it:\n"
         + R.read_text(os.path.join(ddir, names[-1]))[:1500])  # fmt: skip
    return None if p in used else p





def next_id(ms):
    return f"W{max([int(m['id'][1:]) for m in ms], default=0) + 1}"


def fill(trans, premises, counterexample=None, must=None):
    """Write one member per premise, in parallel (a premise rejected before restarts from its best attempt).
    (admitted, rejected, cost)."""
    ms = members(active_only=False)
    warm = {m["premise"]: m["best"] for m in ms if m["status"] == "rejected" and m.get("best") and not counterexample}
    ids = []
    for _ in premises:
        ids.append(next_id(ms + [{"id": i} for i in ids]))

    def one(a):
        try:
            return synthesize(a[0], a[1], trans, counterexample, must, warm.get(a[1]))
        except Exception as exc:  # one candidate's failure must not discard the others
            return None, 0.0, [f"failed: {type(exc).__name__}: {exc}"[:200]], None

    with ThreadPoolExecutor(max(1, len(premises))) as pool:
        res = list(pool.map(one, zip(ids, premises)))
    admitted, rejected, cost = [], [], 0.0
    for (m, c, log, best), mid, premise in zip(res, ids, premises):
        cost += c
        if m:
            admitted.append(m)
        else:
            rejected.append({"id": mid, "premise": premise, "status": "rejected", "born": R.last_exp(), "log": log, "best": best})
    keep = [m for m in members(active_only=False) if not (m["status"] == "rejected" and m["premise"] in {r["premise"] for r in rejected})]
    R.write_json(MEMBERS_FILE, keep + admitted + rejected)  # only the newest rejection per premise is kept
    return admitted, rejected, cost


def maintain(trans):
    """Re-verify members on all experiments and retire those that no longer replay; on a counterexample (a new
    result every member fails, beyond noise), write CEGIS candidates that must replay it and let them replace
    the weakest members; renew a collapsed committee; fill up to K, backing off when no candidate is admitted.
    (cost, notes)."""
    cost, notes, st = 0.0, [], state()
    ms = members(active_only=False)
    last_checked = st.get("last_checked", "")
    fresh = [tr for tr in trans if R.exp_num(tr["exp"]) > R.exp_num(last_checked)] if last_checked else trans[-1:]
    latest = None  # the newest fresh transition that every member fails (all of them are recorded)
    before = {m["id"] for m in ms if m["status"] == "active"}
    fails = {tr["exp"]: set() for tr in fresh}
    refuted = {tr["exp"]: [] for tr in fresh}
    for m in ms:
        if m["status"] != "active":
            continue
        passes, mism, err = replay(source_of(m["id"]), trans)
        rate = sum(passes) / len(passes) if passes else 0.0
        m["replay"] = round(rate, 3)
        for ok, tr in zip(passes, trans):
            if tr["exp"] in fails and not ok:
                fails[tr["exp"]].add(m["id"])
                refuted[tr["exp"]] += [f"  {m['id']} predicted: {x.split(': ', 1)[-1]}" for x in mism if x.startswith(tr["exp"] + " ")]
        if err or rate < ADMIT:
            m["status"], m["retired"] = "retired", f"after {R.last_exp()}: replays {rate:.0%}" + (f" ({err[:80]})" if err else "")
            notes.append(f"retired {m['id']} (replays {rate:.0%})")
    R.write_json(MEMBERS_FILE, ms)
    opened = [e for e in st.get("open_counterexamples", [])
              if not any(replay(source_of(m["id"]), [tr for tr in trans if tr["exp"] == e])[0] == [True] for m in members())]  # fmt: skip
    cexs = [tr for tr in fresh if before and fails[tr["exp"]] == before]  # results that refute every member, beyond noise
    if trans:
        set_state(last_checked=trans[-1]["exp"])
    if cexs:
        latest = cexs[-1]
        opened += [tr["exp"] for tr in cexs]
        set_state(counterexamples=st.get("counterexamples", []) + [tr["exp"] for tr in cexs])
        notes.append(f"COUNTEREXAMPLE {', '.join(tr['exp'] for tr in cexs)}: no verified world model replays it")
        text = (f"{latest['exp']} (compared with {latest['parent']}): measured {latest['class']}, {json.dumps(latest['measured'])}; "
                f"its settings: {json.dumps(plain(configs()[latest['exp']]))}\nWhat the refuted world models predicted:\n"
                + "\n".join(refuted[latest["exp"]][:8]))  # fmt: skip
        admitted, rejected, c = fill(trans, premises_for(CEGIS_N, set()), counterexample=text, must=latest["exp"])
        cost += c
        active = sorted(members(), key=lambda m: m.get("replay", 0))
        drop = [m["id"] for m in active if m["id"] not in {a["id"] for a in admitted}][: max(0, len(active) - K())]
        ms = members(active_only=False)
        for m in ms:
            if m["id"] in drop:
                m["status"], m["retired"] = "retired", f"replaced by a model written around counterexample {latest['exp']}"
        R.write_json(MEMBERS_FILE, ms)
        notes.append(f"wrote {CEGIS_N} around {latest['exp']}: admitted {', '.join(m['id'] for m in admitted) or 'none'}"
                     + (f", replacing {', '.join(drop)}" if drop else ""))  # fmt: skip
    elif st.get("collapse_streak", 0) >= COLLAPSE_ROUNDS and len(members()) >= 3:
        victim = sorted(members(), key=lambda m: m.get("replay", 0))[0]["id"]
        for m in ms:
            if m["id"] == victim:
                m["status"], m["retired"] = "retired", "the committee collapsed (members voted alike): renewed"
        R.write_json(MEMBERS_FILE, ms)
        set_state(collapse_streak=0)
        notes.append(f"renewed {victim} (members voted alike for {COLLAPSE_ROUNDS} rounds)")
    set_state(open_counterexamples=sorted(set(opened)))
    need = K() - len(members())
    if need > 0:
        if len(trans) < state().get("fill_after", 0):
            notes.append(f"{need} seat(s) empty: writing again once {state()['fill_after']} experiments can be replayed")
        else:
            admitted, rejected, c = fill(trans, premises_for(need, {m["premise"] for m in members()}))
            cost += c
            if not admitted:
                set_state(fill_after=len(trans) + max(3, len(trans) // 3))  # back off instead of rewriting every round
            notes.append(f"wrote {need}: admitted {', '.join(m['id'] for m in admitted) or 'none'}"
                         + (f", rejected {', '.join(r['id'] for r in rejected)}" if rejected else ""))  # fmt: skip
    return cost, notes


# ---------------------------------------------------------------------------
# a round
# ---------------------------------------------------------------------------


def entropy(counts, n):
    """Normalised entropy of a vote over the outcome classes: 1 = as split as n voters can be over 3 classes."""
    if n < 2 or len([c for c in counts if c]) < 2:
        return 0.0
    return -sum(c / n * math.log(c / n) for c in counts if c) / math.log(min(n, len(CLASSES)))


def vote(ar, t, preds_by_member, tip_preds):
    """({member: class}, class distribution, disagreement) for one idea."""
    votes = {}
    for mid, p in preds_by_member.items():
        if mid in tip_preds and "__error__" not in p:
            c = classify(ar, t, p, tip_preds[mid])
            if c:
                votes[mid] = c
    n = len(votes)
    counts = [list(votes.values()).count(c) for c in CLASSES]
    dist = {c: k / n for c, k in zip(CLASSES, counts) if k} if n else {}
    return votes, dist, entropy(counts, n)


def unexplored(cfg, tip_cfg, trans):
    """Settings an idea moves that never varied across the replayed experiments: no member has evidence on
    them, so the members' agreement there is a shared blind spot, not knowledge."""
    cfg_all, seen = configs(), {}
    for tr in trans:
        for c in (cfg_all.get(tr["exp"]), cfg_all.get(tr["parent"])):
            for k, v in plain(c).items():
                seen.setdefault(k, set()).add(json.dumps(v))
    return sorted(k for k, v in plain(cfg).items() if plain(tip_cfg).get(k) != v and len(seen.get(k, ())) < 2)


def forecast(ar, t, idea_cfgs, tip_key, tip_cfg):
    """Every verified member predicts every idea and the best they were described against; votes recorded."""
    if not tip_cfg:
        return {}
    preds = {}
    for m in members():
        out, _ = execute(source_of(m["id"]), {**idea_cfgs, "__tip__": tip_cfg})
        if out:
            preds[m["id"]] = out
    tip_preds = {mid: p["__tip__"] for mid, p in preds.items() if "__error__" not in p.get("__tip__", {"__error__": 1})}
    obj, now, rows, stats = t["objective"]["metric"], time.time(), [], {}
    trans = transitions()
    for iid in idea_cfgs:
        votes, dist, dis = vote(ar, t, {mid: p.get(iid, {"__error__": 1}) for mid, p in preds.items()}, tip_preds)
        if not votes:
            continue
        vals = [preds[mid][iid][obj] for mid in votes if isinstance(preds[mid][iid].get(obj), float)]
        stats[iid] = {"votes": votes, "dist": {k: round(v, 3) for k, v in dist.items()}, "plurality": max(dist, key=dist.get),
                      "disagreement": round(dis, 3), "n": len(votes), "unexplored": unexplored(idea_cfgs[iid], tip_cfg, trans),
                      "objective_range": [round(min(vals), 4), round(max(vals), 4)] if vals else None}  # fmt: skip
        rows.append({"ts": now, "after": R.last_exp(), "tip": tip_key, "idea": iid, **stats[iid],
                     "preds": {mid: preds[mid][iid] for mid in votes}})  # fmt: skip
    append_jsonl(FORECASTS_FILE, rows)
    return stats


def act(stats, kind):
    """Raise the most disputed idea (while disagreement is measured to predict errors); drop an idea every
    member says misses a constraint, running every VETO_AUDIT_EVERY-th as an audit."""
    done, st, trusted = [], state(), disagreement_trusted()
    with R.locked():
        ideas = R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS)
        for i in ideas:
            s = stats.get(i["id"])
            if not s or i["status"] != "queued" or i["id"] in st.get("audits", []):
                continue
            if s["n"] >= 4 and s["dist"].get("infeasible") == 1.0 and not s.get("unexplored"):
                st["vetoes"] = st.get("vetoes", 0) + 1
                if st["vetoes"] % VETO_AUDIT_EVERY == 0:
                    st.setdefault("audits", []).append(i["id"])
                    i["note"] = (i["note"] + " | " if i["note"] else "") + "committee veto AUDIT: run it to measure the veto"
                    done.append(f"veto audit {i['id']}")
                else:
                    i["status"], i["note"] = "dropped", f"committee veto: all {s['n']} verified world models predict it misses a constraint"
                    done.append(f"vetoed {i['id']}")
        if kind == "update" and trusted:
            live = [(info(s), iid) for iid, s in stats.items()
                    if any(i["id"] == iid and i["status"] == "queued" for i in ideas)]  # fmt: skip
            if live:
                d, iid = max(live)
                for i in ideas:
                    if d >= 0.5 and i["id"] == iid and int(i["priority"] or 0) < 5 and "most disputed" not in i["note"]:
                        i["priority"] = str(int(i["priority"] or 0) + 1)
                        i["note"] = (i["note"] + " | " if i["note"] else "") + f"committee: most disputed (disagreement {d:.2f}), +1 priority"
                        done.append(f"raised {iid} (disagreement {d:.2f})")
        R.write_tsv(R.IDEAS_FILE, R.IDEA_COLS, ideas)
    set_state(vetoes=st.get("vetoes", 0), audits=st.get("audits", []))
    return done


def run_round(kind, ideas_filter=None):
    """kind 'update' (after an experiment) or 'forecast' (new ideas only). Returns a summary."""
    os.makedirs(DIR, exist_ok=True)
    ar, t = load_ar(), task()
    cost, notes, actions, scored = 0.0, [], [], []
    if kind == "update":
        _, c = featurize_new_runs()
        cost += c
        scored = score_new()
        notes += [f"{s['exp']}: no member's vote predicted '{s['actual']}'" for s in scored if s["refuted_all"]]
    trans = transitions()
    if kind == "update" and len(trans) >= MIN_TRANSITIONS:
        c, n2 = maintain(trans)
        cost += c
        notes += n2
    queued = [i for i in R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS) if i["status"] == "queued"]
    if ideas_filter is not None:
        queued = [i for i in queued if i["id"] in ideas_filter]
    stats = {}
    if queued and members():
        tip_key, tip_cfg = current_tip()  # read once: a keep logged meanwhile must not change the baseline mid-vote
        idea_cfgs, c = featurize_queue(queued, tip_key, tip_cfg)
        cost += c
        stats = forecast(ar, t, idea_cfgs, tip_key, tip_cfg) if idea_cfgs else {}
        actions = act(stats, kind) if stats else []
    if kind == "update":
        distinct = len({tuple(s["votes"].get(m["id"]) for s in stats.values()) for m in members()}) if stats else 0
        collapsed = bool(stats) and len(members()) >= 3 and distinct <= 1
        set_state(collapse_streak=(state().get("collapse_streak", 0) + 1) if collapsed else 0)
    rnd = len(read_jsonl(ROUNDS_FILE)) + 1
    append_jsonl(ROUNDS_FILE, [{"round": rnd, "ts": time.time(), "after": R.last_exp(), "kind": kind, "cost": round(cost, 4),
                                "members": [m["id"] for m in members()], "stats": stats, "actions": actions,
                                "notes": notes, "scored": [s["exp"] for s in scored]}])  # fmt: skip
    write_report()
    summary = (f"round {rnd} ({kind}): {len(members())}/{K()} verified members on {len(trans)} experiments, "
               f"{len(stats)} ideas voted" + (f"; {'; '.join(notes)}" if notes else "")
               + (f"; {'; '.join(actions)}" if actions else ""))  # fmt: skip
    R.journal("committee", cost, summary[:900])
    return summary


# ---------------------------------------------------------------------------
# scoring and calibration
# ---------------------------------------------------------------------------


def score_new():
    """Ledger rows for experiments whose idea was voted on before it started, against the same code it was then
    compared with (a vote cast against an older best is not scored). Each row is one ACI step."""
    done = {r["exp"] for r in read_jsonl(LEDGER_FILE)}
    fcs, st, new = read_jsonl(FORECASTS_FILE), state(), []
    rows = R.results()
    for idx, r in enumerate(rows):
        if r["exp"] in done or r["status"] == "crash" or r.get("tag", "-") in ("", "-"):
            continue
        parent = tip_before(rows, idx)
        run = R.read_json(os.path.join(R.RUNS_DIR, f"{r['exp']}.json"), {}) or {}
        prior = [f for f in fcs if f["idea"] == r["tag"] and f["ts"] < run.get("started", 0)
                 and parent and f.get("tip") == parent["exp"]]  # fmt: skip
        if not prior:
            continue
        f, actual = prior[-1], outcome_class(r)
        share = f["dist"].get(actual, 0.0)
        # adaptive conformal inference over the vote: score = 1 - share of the realised outcome
        alpha, past = st.get("aci_alpha", 0.1), st.get("aci_past", [])
        if past:
            srt = sorted(past)
            j = min(len(srt) - 1, int(math.ceil(min(1.0, max(0.0, 1 - alpha)) * (len(srt) + 1))) - 1)
            q = srt[max(0, j)]
        else:
            q = 1.0
        abstain = q >= 1.0
        hit = abstain or (1 - share) <= q
        st["aci_past"], st["aci_alpha"] = past + [1 - share], alpha + 0.05 * (0.1 - (0 if hit else 1))
        new.append({"exp": r["exp"], "idea": r["tag"], "actual": actual, "plurality": f["plurality"],
                    "correct": f["plurality"] == actual, "disagreement": f["disagreement"], "share": share,
                    "unexplored": f.get("unexplored", []), "refuted_all": share == 0.0, "n": f["n"],
                    "aci_hit": hit, "aci_abstain": abstain, "audit": r["tag"] in st.get("audits", [])})  # fmt: skip
    set_state(aci_past=st.get("aci_past", []), aci_alpha=st.get("aci_alpha", 0.1))
    append_jsonl(LEDGER_FILE, new)
    return new


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def _rate(xs):
    return sum(xs) / len(xs) if xs else None


def health():
    led = read_jsonl(LEDGER_FILE)
    out = {"scored": len(led), "members": len(members()), "k": K(), "transitions": len(transitions())}
    if led:
        out["vote_right"] = _rate([r["correct"] for r in led])
        out["auroc"] = auroc([r["disagreement"] for r in led], [not r["correct"] for r in led])
        out["unanimous_error"] = _rate([not r["correct"] for r in led if r["disagreement"] == 0])
        out["split_error"] = _rate([not r["correct"] for r in led if r["disagreement"] > 0])
        out["aci_coverage"] = _rate([r["aci_hit"] for r in led])
        out["aci_abstain"] = _rate([r["aci_abstain"] for r in led])
        out["refuted_all"] = sum(r["refuted_all"] for r in led)
        out["audits"] = [(r["exp"], r["actual"]) for r in led if r["audit"]]
    return out


def disagreement_trusted(h=None):
    h = h or health()
    return h.get("scored", 0) < AUROC_N or h.get("auroc") is None or h["auroc"] >= AUROC_MIN


def fmt_health(h=None):
    h = h or health()
    parts = [f"{h['members']}/{h['k']} verified world models on {h['transitions']} experiments"]
    if h.get("scored"):
        parts.append(f"{h['scored']} votes scored, plurality right {h['vote_right']:.0%}")
        if h.get("auroc") is not None:
            parts.append(f"disagreement predicts vote errors: AUROC {h['auroc']:.2f}"
                         + ("" if disagreement_trusted(h) else " (not informative: priorities ignore it)"))
        if h.get("unanimous_error") is not None and h.get("split_error") is not None:
            parts.append(f"error when unanimous {h['unanimous_error']:.0%} vs split {h['split_error']:.0%}")
        parts.append(f"conformal coverage {h['aci_coverage']:.0%} (target 90%), abstained {h['aci_abstain']:.0%}")
        parts.append(f"{h['refuted_all']} outcomes no member voted for")
    elif h["transitions"] < MIN_TRANSITIONS:
        parts.append(f"members are written once {MIN_TRANSITIONS} experiments can be replayed")
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# what the rest of the framework reads
# ---------------------------------------------------------------------------


def latest_round(kind=None):
    rs = [r for r in read_jsonl(ROUNDS_FILE) if kind is None or r["kind"] == kind]
    return rs[-1] if rs else None


def current_forecasts():
    """{idea: newest vote}."""
    out = {}
    for f in read_jsonl(FORECASTS_FILE):
        out[f["idea"]] = f
    return out


def revise_reasons(last_revise_exp):
    evs = [e for e in state().get("counterexamples", []) if R.exp_num(e) > last_revise_exp]
    if evs and R.exp_num(R.last_exp()) - max(last_revise_exp, 0) >= REVISE_GAP:
        return [f"counterexample(s) {', '.join(evs)}: results no verified world model replays (a mechanism is missing)"]
    return []


def mechanisms_text():
    return "\n".join(f"  {m['id']} (replays {m.get('replay', 0):.0%}; premise: {m['premise'][:90]}): {m['mechanisms']}"
                     for m in members())  # fmt: skip


def disputed(n=3):
    """Queued ideas by how much running them would teach: outside the evidence first, then by disagreement."""
    fc, ideas = current_forecasts(), {i["id"]: i for i in R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS)}
    live = [(f["disagreement"], iid, f) for iid, f in fc.items() if ideas.get(iid, {}).get("status") == "queued"]
    return sorted(live, key=lambda x: (-info(x[2]), -x[0]))[:n]


def info(f):
    return 1.0 if f.get("unexplored") else f["disagreement"]


def dr_question():
    """The literature question the committee most needs answered: the newest counterexample, else the most
    informative queued idea not asked about with the same vote before; None if neither (the literature step
    then picks its own question)."""
    st = state()
    evs = st.get("counterexamples", [])
    if evs and f"cex:{evs[-1]}" not in st.get("asked", []):
        set_state(asked=sorted(set(st.get("asked", [])) | {f"cex:{evs[-1]}"}))
        tr = next((x for x in transitions() if x["exp"] == evs[-1]), None)
        desc = next((r["description"] for r in R.results() if r["exp"] == evs[-1]), "")
        if tr:
            return (f"What mechanism explains this result, which none of our verified world models predicted? "
                    f"{tr['exp']}: {desc[:400]} Outcome: {tr['class']}, {json.dumps(tr['measured'])}. "
                    f"Our world models believe:\n{mechanisms_text()[:1500]}")  # fmt: skip
    asked = set(st.get("asked", []))
    ideas = {i["id"]: i for i in R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS)}
    for _, iid, f in disputed(6):
        key = f"{iid}:{json.dumps(f['dist'], sort_keys=True)}:{','.join(f.get('unexplored') or [])}"
        if info(f) < 0.5 or key in asked:
            continue
        set_state(asked=sorted(asked | {key}))
        desc = ideas.get(iid, {}).get("description", "")[:400]
        if f.get("unexplored"):
            return (f"None of our experiments has varied {', '.join(f['unexplored'])}, so our world models have no evidence "
                    f"on this proposed change: {desc} What does the literature say about its effect in settings like "
                    f"ours, and what mechanism decides it? Our world models believe:\n{mechanisms_text()[:1500]}")  # fmt: skip
        split = ", ".join(f"{mid}: {c}" for mid, c in f["votes"].items())
        return (f"World models that all fit our results disagree on this proposed change ({split}): {desc} "
                f"What does the literature say about its effect, and which mechanism decides it? "
                f"The world models believe:\n{mechanisms_text()[:1500]}")  # fmt: skip
    return None


def lit_question():
    """For the triggered literature mode: a question only for a counterexample not asked about yet."""
    st = state()
    evs = st.get("counterexamples", [])
    return dr_question() if evs and f"cex:{evs[-1]}" not in st.get("asked", []) else None


def inputs_for(role):
    if not members() and not read_jsonl(LEDGER_FILE):
        return ""
    lines = ["=== COMMITTEE OF VERIFIED WORLD MODELS (programs written independently, each replaying the results) ===",
             fmt_health(), "Their mechanisms (competing explanations that all fit the data):",
             mechanisms_text() or "  (none admitted yet)"]  # fmt: skip
    evs = state().get("counterexamples", [])[-5:]
    if evs:
        lines.append(f"Counterexamples (results no verified world model replayed): {', '.join(evs)}")
    if role == "ideate":
        lines.append("Their votes on the queued ideas (improved / not-improved / infeasible; disagreement 0-1):")
        lines += [f"  {iid}: {f['dist']} disagreement {d:.2f}" + (f"; OUTSIDE THE EVIDENCE (moves never-varied settings: "
                  f"{', '.join(f['unexplored'])}), so the agreement is not knowledge" if f.get("unexplored") else "")
                  for d, iid, f in disputed(12)]  # fmt: skip
    return "\n".join(lines) + "\n"


def write_report():
    lines = ["# Committee of verified world models", "", fmt_health(), "", "## Members"]
    for m in members(active_only=False):
        if m["status"] != "rejected":
            lines.append(f"- {m['id']} {m['status']} (since {m.get('born')}, replays {m.get('replay', 0):.0%}): "
                         f"{m.get('mechanisms', '')}" + (f" [retired: {m['retired']}]" if m.get("retired") else ""))  # fmt: skip
    rej = [m for m in members(active_only=False) if m["status"] == "rejected"]
    if rej:
        lines.append(f"\n{len(rej)} candidate(s) rejected for not replaying the results.")
    lines += ["", "## Votes on the queue", "idea | votes | disagreement | objective range | outside the evidence", "---|---|---|---|---"]
    lines += [f"{iid} | {f['dist']} | {d:.2f} | {f.get('objective_range')} | {', '.join(f.get('unexplored') or []) or '-'}"
              for d, iid, f in disputed(20)]  # fmt: skip
    evs = state().get("counterexamples", [])
    if evs:
        lines += ["", "## Counterexamples", ", ".join(evs)]
    os.makedirs(DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def status_lines():
    return [f"committee: {fmt_health()}"] if latest_round() else ["committee: no round yet"]


def after_seal(idea, started):
    prior = [f for f in read_jsonl(FORECASTS_FILE) if f["idea"] == idea and f["ts"] < started]
    if not prior:
        return []
    f = prior[-1]
    return [f"committee vote for {idea} (made before launch): {f['dist']}, disagreement {f['disagreement']:.2f}, "
            f"objective range {f.get('objective_range')}"] + [f"  {mid}: {c}" for mid, c in f["votes"].items()]  # fmt: skip
