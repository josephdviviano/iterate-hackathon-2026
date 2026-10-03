"""Frame-aware synthesis and verification, mirroring OPINE-World's input.

OPINE's transition rule takes the rendered frame and reads walls, floor and
hazards from grid cells; the object extractor is a side view. Here the
program gets both: `transition_function(state, action, frame)` where `frame`
is the 64x64 before grid. The output stays the object list, so the
committee's canonical comparison and detectors apply unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from committee.loader import Transition
from committee.synth import CONTRACT, render_transitions
from committee.verify import Verdict, states_equal, static_violations

FRAME_CONTRACT = """
# Frame input

`transition_function(state, action, frame)` takes a third argument: the
64x64 before frame as a list of 64 rows of 64 colour indices (0 to 15),
`frame[y][x]`. The object extractor does not emit walls, floor, hazards or
teleporter pads; they are visible in the frame. Read geometry from the
frame as a rule over cell colours, for example "a move is blocked when the
cell ahead has the wall colour". Do not list coordinates of observed
positions. Return the object list only, not a frame.
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
        pred = json.loads(json.dumps(mod.transition_function(copy.deepcopy(r["before"]), r["action"], copy.deepcopy(r["frame"]))))
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


def transition_function(state, action, frame):
    return state
'''

_RUNNER = r'''
import sys, json, importlib.util, copy, traceback
sys.setrecursionlimit(10000)
spec = importlib.util.spec_from_file_location("program", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    json.dump({"load_error": traceback.format_exc()[-800:]}, open(sys.argv[3], "w")); sys.exit(0)
rows = json.load(open(sys.argv[2]))
out = []
for r in rows:
    try:
        pred = mod.transition_function(copy.deepcopy(r["before"]), r["action"], copy.deepcopy(r["frame"]))
        out.append({"pred": json.loads(json.dumps(pred))})
    except Exception:
        out.append({"error": traceback.format_exc()[-400:]})
json.dump(out, open(sys.argv[3], "w"))
'''

HEX = "0123456789abcdef"


def render_frame(grid: list[list[int]]) -> str:
    return "\n".join("".join(HEX[v] for v in row) for row in grid)


def changed_cells(a: list[list[int]], b: list[list[int]]) -> list[tuple[int, int, int, int]]:
    return [(x, y, a[y][x], b[y][x]) for y in range(len(a)) for x in range(len(a[0])) if a[y][x] != b[y][x]]


def render_frames(train: list[Transition]) -> str:
    out = ["\n# Frames\n", "Initial before frame, one hex digit per cell, row y from top, column x from left:\n",
           "```\n" + render_frame(train[0].before_grid) + "\n```\n",
           "Cells that change per transition (x, y, before, after), up to 12 shown:\n"]
    for i, t in enumerate(train):
        cells = changed_cells(t.before_grid, t.after_grid)
        shown = ", ".join(f"({x},{y},{HEX[a]}>{HEX[b]})" for x, y, a, b in cells[:12])
        out.append(f"- transition {i}: {len(cells)} cells changed" + (f": {shown}" if cells else ""))
    return "\n".join(out) + "\n"


def write_workspace(ws: Path, train: list[Transition], extra_task: str = "") -> None:
    (ws / "TASK.md").write_text(CONTRACT + FRAME_CONTRACT + extra_task)
    (ws / "transitions.md").write_text(render_transitions(train) + render_frames(train))
    (ws / "buffer.json").write_text(json.dumps(
        [{"step": t.step, "action": t.action, "before": t.before_objs, "after": t.after_objs, "frame": t.before_grid}
         for t in train]))
    (ws / "check.py").write_text(CHECK_SCRIPT)
    (ws / "program.py").write_text(STUB)


def run_program(source: str, train: list[Transition], test: list[Transition], timeout_s: float = 120) -> Verdict:
    bad = static_violations(source)
    if bad:
        return Verdict([], [], [], error=f"forbidden: {bad}")
    rows = [{"before": t.before_objs, "action": t.action, "frame": t.before_grid} for t in train + test]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "program.py").write_text(source)
        (p / "run.py").write_text(_RUNNER)
        (p / "rows.json").write_text(json.dumps(rows))
        try:
            proc = subprocess.run([sys.executable, "-I", "run.py", "program.py", "rows.json", "out.json"],
                                  cwd=tmp, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return Verdict([], [], [], error="timeout")
        if not (p / "out.json").exists():
            return Verdict([], [], [], error=f"crash: {proc.stderr[-400:]}")
        out = json.loads((p / "out.json").read_text())
    if isinstance(out, dict):
        return Verdict([], [], [], error=out["load_error"])
    results = [(r.get("pred"), r.get("pred") is not None and isinstance(r.get("pred"), list)
                and states_equal(r["pred"], t.after_objs)) for r, t in zip(out, train + test)]
    n = len(train)
    return Verdict([ok for _, ok in results[:n]], [ok for _, ok in results[n:]], [pred for pred, _ in results[n:]])
