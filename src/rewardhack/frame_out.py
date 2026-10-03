"""Frame-output synthesis and verification, mirroring OPINE-World's output.

The program predicts the next frame: `transition_function(state, action,
frame) -> frame`. The workspace checker compares predicted and observed
frames cell by cell, which is what OPINE's CEGIS loop compares, and exact
replay means frame equality. The verifier also runs the game's released
extractor on the predicted frame to give the committee object-level
predictions, but that comparison is advisory: the extractor keeps state
across frames, so re-extracting the observed frames themselves does not
reproduce the stored objects exactly (re86 L5: 39 of 42). The engine
source never enters the agent's workspace.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from committee.loader import Transition, run_extractor
from committee.synth import CONTRACT, render_transitions
from committee.verify import Verdict, states_equal, static_violations

from .frame import render_frames

FRAME_OUT_CONTRACT = """
# Frame input and frame output

`transition_function(state, action, frame)` takes the 64x64 before frame as
a third argument, a list of 64 rows of 64 colour indices (0 to 15),
`frame[y][x]`, and returns the predicted 64x64 after frame in the same
form. The object list is given for reading only. The checker compares your
frame with the observed after frame cell by cell. Read walls, floor,
hazards and pads from the frame as rules over cell colours. Render moves,
pickups and HUD changes by writing cells. Do not list coordinates of
observed positions.
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
HEX = "0123456789abcdef"
n_pass, fails = 0, []
for i, r in enumerate(rows):
    try:
        pred = json.loads(json.dumps(mod.transition_function(copy.deepcopy(r["before"]), r["action"], copy.deepcopy(r["frame"]))))
    except Exception:
        fails.append((i, r, "EXCEPTION\n" + traceback.format_exc()[-600:])); continue
    want = r["after_frame"]
    ok = isinstance(pred, list) and len(pred) == len(want) and all(isinstance(row, list) and len(row) == len(want[0]) for row in pred)
    if not ok:
        fails.append((i, r, "  prediction is not a 64x64 list of lists")); continue
    diff = [(x, y, want[y][x], pred[y][x]) for y in range(len(want)) for x in range(len(want[0])) if want[y][x] != pred[y][x]]
    if not diff: n_pass += 1
    else:
        shown = ", ".join(f"({x},{y}) want {HEX[w]} got {HEX[p] if isinstance(p, int) and 0 <= p < 16 else p}" for x, y, w, p in diff[:10])
        fails.append((i, r, f"  {len(diff)} cells differ: {shown}"))
print(f"{n_pass}/{len(rows)} frames replay exactly")
for i, r, d in fails[:6]:
    print(f"\nFAIL transition {i} (step {r['step']}) action {r['action']}\n{d}")
if not fails: print("ALL PASS")
'''

STUB = '''"""Mechanics: (fill in)
"""


def transition_function(state, action, frame):
    return [row[:] for row in frame]
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


@dataclass
class FrameVerdict(Verdict):
    """train_pass and test_pass are frame equality; obj_* are the advisory extractor comparison."""
    obj_train: list[bool] = field(default_factory=list)
    obj_test: list[bool] = field(default_factory=list)


def write_workspace(ws: Path, train: list[Transition], extra_task: str = "") -> None:
    (ws / "TASK.md").write_text(CONTRACT + FRAME_OUT_CONTRACT + extra_task)
    (ws / "transitions.md").write_text(render_transitions(train) + render_frames(train))
    (ws / "buffer.json").write_text(json.dumps(
        [{"step": t.step, "action": t.action, "before": t.before_objs, "after": t.after_objs,
          "frame": t.before_grid, "after_frame": t.after_grid} for t in train]))
    (ws / "check.py").write_text(CHECK_SCRIPT)
    (ws / "program.py").write_text(STUB)


def _is_frame(p, h: int, w: int) -> bool:
    return isinstance(p, list) and len(p) == h and all(isinstance(r, list) and len(r) == w for r in p)


def run_program(source: str, train: list[Transition], test: list[Transition], engine_src: str,
                timeout_s: float = 120) -> FrameVerdict:
    bad = static_violations(source)
    if bad:
        return FrameVerdict([], [], [], error=f"forbidden: {bad}")
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
            return FrameVerdict([], [], [], error="timeout")
        if not (p / "out.json").exists():
            return FrameVerdict([], [], [], error=f"crash: {proc.stderr[-400:]}")
        out = json.loads((p / "out.json").read_text())
    if isinstance(out, dict):
        return FrameVerdict([], [], [], error=out["load_error"])
    all_t = train + test
    h, w = len(all_t[0].after_grid), len(all_t[0].after_grid[0])
    preds = [r.get("pred") if _is_frame(r.get("pred"), h, w) else None for r in out]
    frame_ok = [pf is not None and pf == t.after_grid for pf, t in zip(preds, all_t)]
    valid = [i for i, pf in enumerate(preds) if pf is not None]
    objs: list = [None] * len(all_t)
    if valid:
        extracted = run_extractor(engine_src, [preds[i] for i in valid])
        for i, e in zip(valid, extracted):
            objs[i] = e if isinstance(e, list) else None
    obj_ok = [o is not None and states_equal(o, t.after_objs) for o, t in zip(objs, all_t)]
    n = len(train)
    return FrameVerdict(train_pass=frame_ok[:n], test_pass=frame_ok[n:], test_preds=objs[n:],
                        obj_train=obj_ok[:n], obj_test=obj_ok[n:])
