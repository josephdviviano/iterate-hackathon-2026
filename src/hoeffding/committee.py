"""A committee of strategies for Hoeffding's problem.

A member is a program `strategy(n, m, t) -> (atoms, weights)`. Its value on an
instance is the certified value of its candidate, or None when rejected. MDL
weights come from the ARC committee. On each instance the committee reports
the best certified value (a rigorous lower bound), the Hoeffding wall above
it, and the MDL-weighted disagreement over which member leads.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from committee.committee import description_length

from .problem import Instance, bernoulli_value, hoeffding_bound
from .verify import Certificate, Rejected, certify

_RUNNER = r'''
import sys, os, json, importlib.util, traceback
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
spec = importlib.util.spec_from_file_location("strategy", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    json.dump({"load_error": traceback.format_exc()[-800:]}, open(sys.argv[3], "w")); sys.exit(0)
out = []
for d in json.load(open(sys.argv[2])):
    try:
        n, m, t = int(d["n"]), Fraction(d["m"]), Fraction(d["t"])
        atoms, weights = mod.strategy(n, m, t)
        conf = None
        if hasattr(mod, "confidence"):
            try:
                conf = float(mod.confidence(n, m, t))
            except Exception:
                conf = None
        out.append({"atoms": [str(a) for a in atoms], "weights": [str(w) for w in weights], "confidence": conf})
    except Exception:
        out.append({"error": traceback.format_exc()[-400:]})
json.dump(out, open(sys.argv[3], "w"))
'''

FORBIDDEN = ("open(", "subprocess", "socket", "urllib", "requests", "__import__", "importlib", "eval(", "exec(")


def run_strategy(source: str, instances: list[Instance], timeout_s: float = 300,
                 confidences: list | None = None) -> list[Certificate | str]:
    """Run the member in a subprocess on every instance and certify each candidate here.

    When `confidences` is given, the member's confidence(n, m, t) per instance is appended to it (None if absent)."""
    def fail(msg: str) -> list:
        if confidences is not None:
            confidences.extend([None] * len(instances))
        return [msg] * len(instances)

    bad = [p for p in FORBIDDEN if p in source]
    if bad:
        return fail(f"forbidden: {bad}")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "strategy.py").write_text(source)
        from .synth import _standalone_verifier  # the workspace helper the member was allowed to import
        (p / "verify.py").write_text(_standalone_verifier())
        (p / "run.py").write_text(_RUNNER)
        (p / "inst.json").write_text(json.dumps([i.to_json() for i in instances]))
        try:
            subprocess.run([sys.executable, "-I", "run.py", "strategy.py", "inst.json", "out.json"],
                           cwd=tmp, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return fail("timeout")
        if not (p / "out.json").exists():
            return fail("crash")
        out = json.loads((p / "out.json").read_text())
    if isinstance(out, dict):
        return fail(out["load_error"])
    results = []
    for r, inst in zip(out, instances):
        if confidences is not None:
            confidences.append(r.get("confidence"))
        if "error" in r:
            results.append(r["error"])
            continue
        try:
            results.append(certify(r["atoms"], r["weights"], inst))
        except (Rejected, ValueError, ZeroDivisionError) as e:
            results.append(f"rejected: {e}")
    return results


@dataclass
class Member:
    name: str
    source: str
    results: list[Certificate | str]
    p_tight: list[float | None] = field(default_factory=list)
    length: int = 0

    def value(self, i: int) -> Fraction | None:
        r = self.results[i]
        return r.value if isinstance(r, Certificate) else None


class HoeffdingCommittee:
    def __init__(self, members: list[Member], lam: float = 0.01):
        self.members = members
        self.lam = lam
        for m in members:
            m.length = description_length(m.source)
        base = min(m.length for m in members)
        raw = [math.exp(-lam * (m.length - base)) for m in members]
        z = sum(raw)
        self.weights = [r / z for r in raw]

    def report(self, i: int, inst: Instance) -> dict:
        vals = [m.value(i) for m in self.members]
        certified = [v for v in vals if v is not None]
        best = max(certified) if certified else None
        # leader distribution: MDL weight of each member that attains the best value, else of "rejected"
        dist: dict[str, float] = {}
        for m, w, v in zip(self.members, self.weights, vals):
            key = m.name if v is not None and v == best else ("rejected" if v is None else "behind")
            dist[key] = dist.get(key, 0.0) + w
        h = -sum(p * math.log(p) for p in dist.values() if p > 0)
        disagreement = h / math.log(len(self.members)) if len(self.members) > 1 else 0.0
        wall = hoeffding_bound(inst)
        bern = bernoulli_value(inst)
        return {
            "instance": inst.to_json(),
            "bernoulli": float(bern),
            "certified_lower": float(best) if best is not None else None,
            "hoeffding_upper": wall,
            "unknown_width": (wall - float(best)) if best is not None else (wall - float(bern)),
            "leader_distribution": {k: round(v, 4) for k, v in dist.items()},
            "disagreement": round(disagreement, 4),
            "member_values": [float(v) if v is not None else None for v in vals],
            "member_structures": [len(m.results[i].atoms) if isinstance(m.results[i], Certificate) else None
                                  for m in self.members],
        }

    def reports(self, instances: list[Instance]) -> list[dict]:
        return [self.report(i, inst) for i, inst in enumerate(instances)]
