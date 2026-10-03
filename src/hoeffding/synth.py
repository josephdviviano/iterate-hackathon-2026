"""Synthesize one strategy with the claude CLI in an isolated workspace.

The workspace holds the train instances, the exact certifier, a checker and a
stub. The agent edits strategy.py and writes report.json with, per train
instance, its probability that the certified value is within 1e-4 of the true
supremum. Test instances never enter the workspace.
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

from .problem import Instance, bernoulli_value, hoeffding_bound

CONTRACT = """\
# Task: push a rigorous lower bound on an open extremal probability

Let X_1, ..., X_n be iid on [0, 1] with E X = m, and S_n their sum. For
0 <= t < n m, define p_n(m, t) = sup over such laws of P(S_n <= t). Only n = 1
and n = 2 are solved. Hoeffding's inequality gives the upper wall
exp(-n kl(t/n || m)); the Bernoulli(m) law gives a lower wall P(Bin(n, m) <= t).
Any discrete law you name gives a certified lower bound, computed in exact
rational arithmetic by `verify.py`.

Write `strategy(n, m, t) -> (atoms, weights)` in `strategy.py`:

- `m`, `t` are `fractions.Fraction`. Return two lists: atoms in [0, 1] and
  non-negative weights. Fractions, ints, floats or numeric strings are all
  accepted. Floats are rounded to rationals with denominator <= 10^12.
- The checker re-solves the last two weights so that sum w = 1 and
  sum w a = m hold exactly. If that needs a negative weight, the candidate is
  rejected. Keep atoms distinct.
- Only the Python standard library is available. No file or network access.

Rules:

1. The strategy must be a rule in (n, m, t), not a table of answers. It will be
   run on instances you have not seen.
2. Run `python3 check.py`. It prints, per train instance, the certified value,
   the Bernoulli wall, the Hoeffding wall and the structure of your measure.
   Iterate until you cannot improve. Numerical optimization inside `strategy`
   is allowed; keep each call under 5 seconds.
3. Also define `confidence(n, m, t) -> float` in `strategy.py`: your
   probability, in [0, 1], that the certified value of `strategy(n, m, t)` is
   within 1e-4 of the true supremum p_n(m, t). It is called on unseen
   instances too, so make it a rule of (n, m, t) and of what your search found.
   Be honest: it is scored for calibration, not for optimism. Before you stop,
   also write `report.json` mapping each train key (as printed by the checker)
   to that probability.
4. Put a 5-line header comment in `strategy.py` stating the structure you
   found (how many atoms, where they sit) and what you could not settle.

Files: `instances.json`, `verify.py`, `check.py`, `strategy.py`, `report.json`.
"""

CHECK_SCRIPT = r'''
import json, importlib.util, traceback, sys, math, time
from fractions import Fraction
import verify
rows = json.load(open("instances.json"))
spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    print("LOAD ERROR\n" + traceback.format_exc()[-1500:]); sys.exit(1)
print(f"{'key':24s} {'certified':>10s} {'bernoulli':>10s} {'hoeffding':>10s}  structure")
for d in rows:
    n, m, t = int(d["n"]), Fraction(d["m"]), Fraction(d["t"])
    t0 = time.time()
    try:
        atoms, weights = mod.strategy(n, m, t)
        c = verify.certify_raw(atoms, weights, n, m, t)
        atoms_s = " ".join(f"{float(a):.4f}:{float(w):.3f}" for a, w in zip(c["atoms"], c["weights"]))
        print(f"{d['key']:24s} {c['value']:10.6f} {d['bernoulli']:10.6f} {d['hoeffding']:10.6f}  "
              f"{len(c['atoms'])} atoms [{atoms_s}]{' (repaired)' if c['repaired'] else ''}  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"{d['key']:24s} {'REJECTED':>10s} {d['bernoulli']:10.6f} {d['hoeffding']:10.6f}  {str(e)[-200:]}")
'''

STUB = '''"""Structure: (fill in)
"""
from fractions import Fraction


def strategy(n, m, t):
    # Bernoulli(m): atoms {0, 1}. Replace with a better law.
    return [0, 1], [1 - m, m]


def confidence(n, m, t):
    # Probability that strategy(n, m, t) is within 1e-4 of the supremum. Replace.
    return 0.0
'''


def _standalone_verifier() -> str:
    src = (Path(__file__).parent / "verify.py").read_text()
    src = src.replace("from .problem import Instance\n", "")
    src += '''

def certify_raw(atoms, weights, n, m, t):
    class _I:
        pass
    inst = _I(); inst.n, inst.m, inst.t = n, m, t
    c = certify(atoms, weights, inst)
    return {"atoms": c.atoms, "weights": c.weights, "value": float(c.value), "repaired": c.repaired}
'''
    return src


@dataclass
class SynthResult:
    source: str
    seed: str | None
    report: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    check_output: str = ""


def _write_workspace(ws: Path, train: list[Instance], seed: str | None) -> None:
    task = CONTRACT
    if seed:
        task += "\n# Hypothesis to build on\n\n" + seed.strip() + "\n"
    (ws / "TASK.md").write_text(task)
    (ws / "instances.json").write_text(json.dumps(
        [dict(i.to_json(), key=i.key, bernoulli=float(bernoulli_value(i)), hoeffding=hoeffding_bound(i))
         for i in train], indent=1))
    (ws / "verify.py").write_text(_standalone_verifier())
    (ws / "check.py").write_text(CHECK_SCRIPT)
    (ws / "strategy.py").write_text(STUB)


def synthesize(train: list[Instance], seed: str | None = None, *, model: str = "opus",
               max_turns: int = 40, timeout_s: float = 900, keep_dir: Path | None = None) -> SynthResult:
    return synthesize_in(train, seed, model=model, max_turns=max_turns, timeout_s=timeout_s, keep_dir=keep_dir)


def synthesize_in(train: list[Instance], seed: str | None = None, *, model: str = "opus",
                  max_turns: int = 40, timeout_s: float = 900, keep_dir: Path | None = None) -> SynthResult:
    ws = Path(tempfile.mkdtemp(prefix="hsynth_"))
    _write_workspace(ws, train, seed)
    prompt = ("Read TASK.md, then implement strategy in strategy.py. Run `python3 check.py` to see certified "
              "values. Stop when you cannot improve, after writing report.json.")
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
                j = next((x for x in j if x.get("type") == "result"), {})
            meta.update({k: j.get(k) for k in ("num_turns", "duration_ms", "total_cost_usd", "is_error", "subtype")})
            meta["final_message"] = (j.get("result") or "")[-1500:]
        except json.JSONDecodeError:
            meta["stdout_tail"] = proc.stdout[-800:]
        meta["stderr_tail"] = proc.stderr[-800:]
        meta["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        meta["timeout"] = True
    meta["wall_s"] = round(time.time() - t0, 1)
    source = (ws / "strategy.py").read_text()
    report = {}
    if (ws / "report.json").exists():
        try:
            report = json.loads((ws / "report.json").read_text())
        except json.JSONDecodeError:
            meta["report_error"] = "invalid json"
    check = subprocess.run(["python3", "check.py"], cwd=ws, capture_output=True, text=True, timeout=300)
    result = SynthResult(source=source, seed=seed, report=report, meta=meta, check_output=check.stdout[-3000:])
    if keep_dir is not None:
        shutil.copytree(ws, keep_dir, dirs_exist_ok=True)
    shutil.rmtree(ws, ignore_errors=True)
    return result


SEEDS = {
    "none": None,
    "binary": "Try two atoms {a, b} with a < m < b. Lemma (Meester 2008): an extremal law has t - a in the "
              "support of S_{n-1}, so t - a = j b + (n - 1 - j) a for an integer j. Enumerate j.",
    "ternary": "Try three atoms {0, a, 1}. Conjecture (Meester 2008): a = (t - l) / k for integers k >= 1, "
               "l >= 0 with l < t < l + k <= n - 1. One weight is free; optimize it.",
    "search": "Treat it as numerical optimization over k <= 4 atoms and weights. Random restarts, then local "
              "moves. Return the best law found. Keep the call under 5 seconds.",
}
