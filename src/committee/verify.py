"""Exact-replay verification of candidate programs in a subprocess.

A program is a Python module that defines
``transition_function(state: list[dict], action) -> list[dict]`` where
``action`` is an int or ``{"action_id": 6, "x": int, "y": int}``. It is run
once, in order, over the train transitions and then the test transitions, so
that a module may carry hidden state across calls. Each call gets the observed
before state, never the module's own prediction. A program passes when every
train prediction equals the observed after state as a multiset of objects.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .loader import Transition

FORBIDDEN = [r"\bopen\s*\(", r"\bpickle\b", r"\bsubprocess\b", r"\bimport\s+socket\b", r"\bfrom\s+socket\s+import\b",
             r"\burllib\b",
             r"\brequests\b", r"\b__import__\b", r"\bimportlib\b", r"\beval\s*\(", r"\bexec\s*\(",
             r"replay", r"buffer\.json"]

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
    before = r["before"] if (not open_loop or prev is None or r.get("resync")) else prev
    try:
        pred = mod.transition_function(copy.deepcopy(before), r["action"])
        pred = json.loads(json.dumps(pred))
        out.append({"pred": pred})
        prev = pred
    except Exception:
        out.append({"error": traceback.format_exc()[-400:]})
        prev = None
json.dump(out, open(sys.argv[3], "w"))
'''


def canonical(state: list[dict]) -> list[str]:
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def states_equal(a: list[dict], b: list[dict]) -> bool:
    return canonical(a) == canonical(b)


def static_violations(source: str) -> list[str]:
    return [p for p in FORBIDDEN if re.search(p, source)]


@dataclass
class Verdict:
    train_pass: list[bool]
    test_pass: list[bool]
    test_preds: list[list[dict] | None]
    error: str | None = None

    @property
    def consistent(self) -> bool:
        return self.error is None and all(self.train_pass)

    @property
    def test_accuracy(self) -> float | None:
        return sum(self.test_pass) / len(self.test_pass) if self.test_pass else None


def run_program(source: str, train: list[Transition], test: list[Transition],
                timeout_s: float = 120, open_loop: bool = False) -> Verdict:
    """Teacher forcing by default. With open_loop, the test rows are predicted from the program's
    own previous prediction; the first test row and any row at a level change start from the
    observed state. Train rows are always teacher forced."""
    bad = static_violations(source)
    if bad:
        return Verdict([], [], [], error=f"forbidden: {bad}")
    rows = [{"before": t.before_objs, "action": t.action, "resync": True} for t in train]
    prev_level = None
    for i, t in enumerate(test):
        rows.append({"before": t.before_objs, "action": t.action,
                     "resync": i == 0 or (prev_level is not None and t.level != prev_level)})
        prev_level = t.level
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "program.py").write_text(source)
        (p / "run.py").write_text(_RUNNER)
        (p / "rows.json").write_text(json.dumps(rows))
        try:
            proc = subprocess.run([sys.executable, "-I", "run.py", "program.py", "rows.json", "out.json",
                                   "open" if open_loop else "teacher"],
                                  cwd=tmp, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return Verdict([], [], [], error="timeout")
        if not (p / "out.json").exists():
            return Verdict([], [], [], error=f"crash: {proc.stderr[-400:]}")
        out = json.loads((p / "out.json").read_text())
    if isinstance(out, dict):
        return Verdict([], [], [], error=out["load_error"])
    results = []
    for r, t in zip(out, train + test):
        pred = r.get("pred")
        results.append((pred, pred is not None and isinstance(pred, list) and states_equal(pred, t.after_objs)))
    n = len(train)
    return Verdict(
        train_pass=[ok for _, ok in results[:n]],
        test_pass=[ok for _, ok in results[n:]],
        test_preds=[pred for pred, _ in results[n:]],
    )


def failure_report(verdict: Verdict, train: list[Transition], limit: int = 5) -> str:
    """Human-readable mismatches on train, for the repair loop."""
    if verdict.error:
        return f"ERROR: {verdict.error}"
    lines = []
    for ok, t in zip(verdict.train_pass, train):
        if ok:
            continue
        lines.append(f"step {t.step} action {t.action}: prediction differs from observed after state")
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "ALL TRAIN TRANSITIONS PASS"


def describe_mismatch(pred: list[dict] | None, actual: list[dict], limit: int = 4) -> str:
    """Objects expected but not predicted, and predicted but not observed, as canonical JSON."""
    if pred is None:
        return "  no prediction (exception)"
    p, a = set(canonical(pred)), set(canonical(actual))
    lines = [f"  expected but not predicted: {m[:300]}" for m in sorted(a - p)[:limit]]
    lines += [f"  predicted but not observed:  {e[:300]}" for e in sorted(p - a)[:limit]]
    return "\n".join(lines)


def train_report(source: str, train: list[Transition], limit: int = 6, timeout_s: float = 120) -> tuple[bool, str]:
    """Run a program on the train transitions and describe the first failures, for a repair loop.
    The program is run as if train were the test set so its predictions come back for the diff."""
    verdict = run_program(source, [], train, timeout_s=timeout_s)
    if verdict.error:
        return False, f"ERROR: {verdict.error}"
    n_pass = sum(verdict.test_pass)
    lines = [f"{n_pass}/{len(train)} transitions replay exactly"]
    if n_pass == len(train):
        return True, lines[0] + "\nALL PASS"
    shown = 0
    for i, (ok, t) in enumerate(zip(verdict.test_pass, train)):
        if ok:
            continue
        lines.append(f"\nFAIL transition {i} (step {t.step}) action {json.dumps(t.action)}")
        lines.append(describe_mismatch(verdict.test_preds[i], t.after_objs))
        shown += 1
        if shown >= limit:
            break
    return False, "\n".join(lines)
