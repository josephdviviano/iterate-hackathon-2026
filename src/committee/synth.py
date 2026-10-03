"""Synthesize one candidate program with the claude CLI in an isolated workspace.

The agent gets a workspace with the train transitions, a program stub and a
checker that reports exact-replay failures. It edits program.py until the
checker passes or the turn budget ends. Test transitions never enter the
workspace. A seed hypothesis, when given, steers the program toward one
explanation so that a committee of runs covers distinct hypotheses.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .loader import Transition
from .matrix import effect_signature, pair_objects

CONTRACT = """\
# Task: write the transition rule of an unknown ARC-AGI-3 game

You see a sequence of observed transitions of one level of a game. Each
transition is (before state, action, after state). A state is a list of object
dicts produced by a fixed extractor; the schema is whatever the objects carry
(name, type, tags, x, y, w, h, layer, visible, pixels, ...).

Write `transition_function(state, action)` in `program.py`:

- `state`: list of object dicts. `action`: an int (1 to 5, 7) or
  `{"action_id": 6, "x": int, "y": int}` for a click at grid cell (x, y).
- Return the predicted after state as a list of object dicts with the same
  schema. Object order does not matter. Every field of every object must match
  the observed after state exactly, including names and pixels.
- The checker calls your function once per transition, in order, always with
  the observed before state. You may keep hidden state across calls, but gate
  it on continuity: if the incoming state is not the state you last returned,
  fall back to a stateless default.

Rules:

1. Model the mechanics. Factor the rule by object type: one update rule per
   type, named guards for interactions between types. Prefer the simplest rule
   that explains every transition and would generalise to unseen layouts.
2. Never tabulate observed states. Do not map a state or step index to its
   successor. Do not read `buffer.json` or any file from `program.py`.
   `open(`, `pickle`, `exec`, `eval`, `importlib`, `subprocess` are rejected.
3. Only the Python standard library is available to `program.py`.
4. Run `python3 check.py` to see which transitions fail and how. Iterate until
   it prints ALL PASS, or until you are convinced the remaining failures need
   observations you do not have. Then stop.
5. Keep `program.py` self-contained and under 400 lines. Put a 5-line header
   comment that states the mechanics you implemented and any hypothesis you
   could not confirm.

Files: `transitions.md` (readable diffs), `buffer.json` (full states),
`program.py` (your code), `check.py` (the checker).
"""

CHECK_SCRIPT = r'''
import json, copy, importlib.util, traceback, sys
rows = json.load(open("buffer.json"))
spec = importlib.util.spec_from_file_location("program", "program.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    print("LOAD ERROR\n" + traceback.format_exc()[-1500:]); sys.exit(1)
def canon(s): return sorted(json.dumps(o, sort_keys=True) for o in s)
def describe(pred, actual):
    P, A = set(canon(pred)), set(canon(actual))
    extra, missing = sorted(P - A), sorted(A - P)
    lines = []
    for m in missing[:4]: lines.append("  expected but not predicted: " + m[:300])
    for e in extra[:4]: lines.append("  predicted but not observed:  " + e[:300])
    return "\n".join(lines)
n_pass, fails = 0, []
for i, r in enumerate(rows):
    try:
        pred = json.loads(json.dumps(mod.transition_function(copy.deepcopy(r["before"]), r["action"])))
    except Exception:
        fails.append((i, r, "EXCEPTION\n" + traceback.format_exc()[-600:])); continue
    if isinstance(pred, list) and canon(pred) == canon(r["after"]): n_pass += 1
    else: fails.append((i, r, describe(pred if isinstance(pred, list) else [], r["after"])))
print(f"{n_pass}/{len(rows)} transitions replay exactly")
for i, r, d in fails[:6]:
    print(f"\nFAIL transition {i} (step {r['step']}) action {r['action']}\n{d}")
if not fails: print("ALL PASS")
'''

STUB = '''"""Mechanics: (fill in)
"""


def transition_function(state, action):
    return state
'''


def render_transitions(train: list[Transition]) -> str:
    out = ["# Observed transitions (in order)\n"]
    out.append("## Initial state\n")
    out.append("```json\n" + json.dumps(train[0].before_objs, sort_keys=True) + "\n```\n")
    for i, t in enumerate(train):
        out.append(f"## transition {i} (step {t.step})  action = {json.dumps(t.action)}\n")
        unchanged = 0
        for bo, ao in pair_objects(t.before_objs, t.after_objs):
            sig = effect_signature(bo, ao)
            if sig == "no_change":
                unchanged += 1
                continue
            if bo is None:
                out.append(f"- born: {json.dumps(ao, sort_keys=True)}")
            elif ao is None:
                out.append(f"- gone: {json.dumps(bo, sort_keys=True)}")
            else:
                changes = {k: [bo.get(k), ao.get(k)] for k in sig.split(",")}
                out.append(f"- {bo.get('name')} ({bo.get('type')}): {json.dumps(changes)}")
        out.append(f"- {unchanged} objects unchanged\n")
    return "\n".join(out)


@dataclass
class SynthResult:
    source: str
    seed: str | None
    meta: dict = field(default_factory=dict)
    check_output: str = ""


def _write_workspace(ws: Path, train: list[Transition], seed: str | None) -> None:
    task = CONTRACT
    if seed:
        task += "\n# Hypothesis to build on\n\n" + seed.strip() + "\n"
    (ws / "TASK.md").write_text(task)
    (ws / "transitions.md").write_text(render_transitions(train))
    (ws / "buffer.json").write_text(json.dumps(
        [{"step": t.step, "action": t.action, "before": t.before_objs, "after": t.after_objs} for t in train]))
    (ws / "check.py").write_text(CHECK_SCRIPT)
    (ws / "program.py").write_text(STUB)


def synthesize(train: list[Transition], seed: str | None = None, *, model: str = "opus",
               max_turns: int = 40, timeout_s: float = 900, keep_dir: Path | None = None) -> SynthResult:
    ws = Path(tempfile.mkdtemp(prefix="synth_"))
    _write_workspace(ws, train, seed)
    prompt = ("Read TASK.md and transitions.md, then implement transition_function in program.py. "
              "Run `python3 check.py` to verify. Stop when it prints ALL PASS or when you cannot "
              "make progress without new observations.")
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--max-turns", str(max_turns), "--no-session-persistence", "--strict-mcp-config",
           "--permission-mode", "acceptEdits",
           "--tools", "Read", "Write", "Edit", "Glob", "Grep", "Bash",
           "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Bash(python3:*)", "Bash(python:*)"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    t0 = time.time()
    meta: dict = {"model": model, "max_turns": max_turns}
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout_s, env=env)
        try:
            j = json.loads(proc.stdout)
            if isinstance(j, list):
                j = next((m for m in j if m.get("type") == "result"), {})
            meta.update({k: j.get(k) for k in ("num_turns", "duration_ms", "total_cost_usd", "is_error", "subtype")})
            meta["final_message"] = (j.get("result") or "")[-1500:]
        except json.JSONDecodeError:
            meta["stdout_tail"] = proc.stdout[-800:]
        meta["stderr_tail"] = proc.stderr[-800:]
        meta["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        meta["timeout"] = True
    meta["wall_s"] = round(time.time() - t0, 1)
    source = (ws / "program.py").read_text()
    check = subprocess.run(["python3", "check.py"], cwd=ws, capture_output=True, text=True, timeout=120)
    result = SynthResult(source=source, seed=seed, meta=meta, check_output=check.stdout[-2000:])
    if keep_dir is not None:
        shutil.copytree(ws, keep_dir, dirs_exist_ok=True)
    shutil.rmtree(ws, ignore_errors=True)
    return result
