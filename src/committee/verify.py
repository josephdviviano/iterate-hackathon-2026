"""Exact-replay verification of candidate programs in a subprocess.

A program is a Python module that defines ``transition_function`` in the
form the mode fixes (see ``env``): ``(state, action)`` or ``(state, action,
frame)``, returning the next object list or, in frame_out, the next frame.
``action`` is an int or ``{"action_id": 6, "x": int, "y": int}``. It is run
once, in order, over the train transitions and then the test transitions, so
that a module may carry hidden state across calls. Each call gets the observed
before state, never the module's own prediction. A program passes when every
train prediction equals the observation: the after state as a multiset of
objects, or in frame_out the after frame cell by cell.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .env import FORBIDDEN, call_expr, extract_objects, frame_diff, is_frame, returns_frame, static_violations, takes_frame
from .loader import Transition

__all__ = ["FORBIDDEN", "Verdict", "canonical", "describe_mismatch", "run_program", "states_equal",
           "static_violations", "train_report"]

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
open_loop = len(sys.argv) > 4 and sys.argv[4] == "open"
out = []
prev = None
for r in rows:
    # open loop: the program consumes its own last prediction, except where the row says to
    # resync (a level entry, or the first row), which mirrors what an agent can observe
    r["before"] = r["before"] if (not open_loop or prev is None or r.get("resync")) else prev
    try:
        pred = CALL
        pred = json.loads(json.dumps(pred))
        out.append({"pred": pred})
        prev = pred
    except Exception:
        out.append({"error": traceback.format_exc()[-400:]})
        prev = None
json.dump(out, open(sys.argv[3], "w"))
'''


def canonical(state) -> list[str]:
    """Order-free key of an object list. A frame (a list of rows) keeps its row order; anything
    else is keyed whole, so a malformed prediction counts as wrong instead of failing."""
    if not isinstance(state, list):
        return [json.dumps(state, sort_keys=True)]
    if state and isinstance(state[0], list):
        return [json.dumps(state)]
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def states_equal(a: list, b: list) -> bool:
    return canonical(a) == canonical(b)


@dataclass
class Verdict:
    train_pass: list[bool]
    test_pass: list[bool]
    test_preds: list[list | None]
    error: str | None = None
    mode: str = "objects"
    test_objs: list[list[dict] | None] | None = None  # frame_out: the extractor's view of each predicted frame

    @property
    def consistent(self) -> bool:
        return self.error is None and all(self.train_pass)

    @property
    def test_accuracy(self) -> float | None:
        return sum(self.test_pass) / len(self.test_pass) if self.test_pass else None


def run_program(source: str, train: list[Transition], test: list[Transition],
                timeout_s: float = 120, open_loop: bool = False, mode: str = "objects",
                engine_src: str | None = None) -> Verdict:
    """Teacher forcing by default. With open_loop (objects mode only), the test rows are predicted
    from the program's own previous prediction; the first test row and any row at a level change
    start from the observed state. Train rows are always teacher forced. In frame_out, engine_src
    is the released engine used to extract objects from the predicted test frames."""
    if open_loop and mode != "objects":
        raise ValueError("open-loop rollouts are defined for the objects mode only")
    bad = static_violations(source)
    if bad:
        return Verdict([], [], [], error=f"forbidden: {bad}", mode=mode)
    all_t = train + test
    rows = [{"before": t.before_objs, "action": t.action, "resync": True} for t in train]
    prev_level = None
    for i, t in enumerate(test):
        rows.append({"before": t.before_objs, "action": t.action,
                     "resync": i == 0 or (prev_level is not None and t.level != prev_level)})
        prev_level = t.level
    if takes_frame(mode):
        for r, t in zip(rows, all_t):
            r["frame"] = t.before_grid
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "program.py").write_text(source)
        (p / "run.py").write_text(_RUNNER.replace("CALL", call_expr(mode)))
        (p / "rows.json").write_text(json.dumps(rows))
        try:
            proc = subprocess.run([sys.executable, "-I", "run.py", "program.py", "rows.json", "out.json",
                                   "open" if open_loop else "teacher"],
                                  cwd=tmp, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return Verdict([], [], [], error="timeout", mode=mode)
        if not (p / "out.json").exists():
            return Verdict([], [], [], error=f"crash: {proc.stderr[-400:]}", mode=mode)
        out = json.loads((p / "out.json").read_text())
    if isinstance(out, dict):
        return Verdict([], [], [], error=out["load_error"], mode=mode)
    if returns_frame(mode):
        preds = [r.get("pred") if is_frame(r.get("pred"), t.before_grid) else None for r, t in zip(out, all_t)]
        ok = [pred is not None and pred == t.after_grid for pred, t in zip(preds, all_t)]
    else:
        preds = [r.get("pred") for r in out]
        ok = [pred is not None and isinstance(pred, list) and states_equal(pred, t.after_objs)
              for pred, t in zip(preds, all_t)]
    n = len(train)
    verdict = Verdict(train_pass=ok[:n], test_pass=ok[n:], test_preds=preds[n:], mode=mode)
    if returns_frame(mode) and engine_src is not None:
        verdict.test_objs = extract_objects(engine_src, preds[n:])
    return verdict


def describe_mismatch(pred: list | None, actual: list, limit: int = 4) -> str:
    """Objects expected but not predicted, and predicted but not observed, as canonical JSON.
    For a frame, the cells that differ."""
    if actual and isinstance(actual[0], list):
        if pred is None:
            return f"  no valid frame: not a {len(actual[0])}x{len(actual)} list of lists of ints, or an exception"
        return frame_diff(pred, actual, limit)
    if pred is None:
        return "  no prediction (exception)"
    p, a = set(canonical(pred)), set(canonical(actual))
    lines = [f"  expected but not predicted: {m[:300]}" for m in sorted(a - p)[:limit]]
    lines += [f"  predicted but not observed:  {e[:300]}" for e in sorted(p - a)[:limit]]
    return "\n".join(lines)


def train_report(source: str, train: list[Transition], limit: int = 6, timeout_s: float = 120,
                 mode: str = "objects") -> tuple[bool, str]:
    """Run a program on the train transitions and describe the first failures, for a repair loop.
    The program is run as if train were the test set so its predictions come back for the diff."""
    verdict = run_program(source, [], train, timeout_s=timeout_s, mode=mode)
    if verdict.error:
        return False, f"ERROR: {verdict.error}"
    n_pass = sum(verdict.test_pass)
    unit = "frames" if returns_frame(mode) else "transitions"
    lines = [f"{n_pass}/{len(train)} {unit} replay exactly"]
    if n_pass == len(train):
        return True, lines[0] + "\nALL PASS"
    shown = 0
    for i, (ok, t) in enumerate(zip(verdict.test_pass, train)):
        if ok:
            continue
        lines.append(f"\nFAIL transition {i} (step {t.step}) action {json.dumps(t.action)}")
        lines.append(describe_mismatch(verdict.test_preds[i], t.after_grid if returns_frame(mode) else t.after_objs))
        shown += 1
        if shown >= limit:
            break
    return False, "\n".join(lines)
