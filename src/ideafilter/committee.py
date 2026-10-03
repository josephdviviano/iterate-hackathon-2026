from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROLES = {
    "mechanism": "You reason from the training dynamics: what the change does to the gradient signal, the effective learning rate, the capacity, the data seen per step. Ignore folklore; argue the mechanism.",
    "empirical": "You reason from the prior findings only: which levers of the same kind helped or hurt in this programme, at what size, and extrapolate. Trust measured deltas over intuition.",
    "cost": "You reason about the time cost: what the change does to the compiled step time, the kernel count, memory traffic and the number of steps. An accuracy gain that costs more time than it saves is a rejection.",
    "skeptic": "You assume most ideas are null at 20 seeds (SE about 0.05 pp) and look for the specific reason this one would be different. Be hard to convince, but not blind to a real mechanism.",
}

CONTRACT = """\
You are one member of a committee that decides whether a proposed change to a
CIFAR-100 speedrun recipe deserves a 20-seed A100 sweep. The score is mean
prepare+train time at a mean top-1 of at least 75.2 percent over 40 fresh seeds.
The recipe is an airbench-lineage convnet trained with SGD, label smoothing,
lookahead EMA, a triangular learning rate and progressive resizing; it reaches
about 75.2 percent in about 5.3 s on the development stack. Seed noise at 20
seeds is about 0.05 pp standard error per arm.

The quantity that matters is prepare+train seconds per trial at the 75.2
percent line. An idea is a POSITIVE if the sweep would show the recipe
reaching 75.2 percent in fewer epochs (about 0.73 s per epoch) or in less
time per epoch, worth at least 2 percent of the score; an accuracy gain at
fixed epochs counts only through the epochs it lets the recipe drop (about
0.25 pp per quarter epoch near 8 epochs). Everything else is NEGATIVE: null,
noise, or a gain that costs more time than it is worth.

Answer with one JSON object and nothing else:
{"p_positive": <0..1>, "pred_dpp": <float, percentage points>, "pred_dtime": <float, fraction, e.g. 0.03 for 3 percent slower>,
 "confidence": <0..1, how sure you are of the sign>, "reason": "<at most 40 words>"}
"""


@dataclass
class Prediction:
    member: str
    p_positive: float
    pred_dpp: float
    pred_dtime: float
    confidence: float
    reason: str
    raw: str = ""


def _ask(prompt: str, model: str, timeout_s: float = 180, system: str = "") -> str:
    """One non-interactive model call with no tools; the system prompt replaces Claude Code's own."""
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--no-session-persistence",
           "--strict-mcp-config", "--disallowedTools", "*", "--max-turns", "1",
           "--system-prompt", system or "You are a forecaster. You answer only with the JSON object requested. You have no tools."]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    try:
        j = json.loads(proc.stdout)
        if isinstance(j, list):
            j = next((x for x in j if x.get("type") == "result"), {})
        return j.get("result") or ""
    except json.JSONDecodeError:
        return proc.stdout


def _parse(member: str, text: str) -> Prediction:
    start, end = text.find("{"), text.rfind("}")
    try:
        d = json.loads(text[start:end + 1])
        return Prediction(member, float(d["p_positive"]), float(d.get("pred_dpp", 0.0)), float(d.get("pred_dtime", 0.0)),
                          float(d.get("confidence", 0.5)), str(d.get("reason", ""))[:300], text[:500])
    except Exception:
        return Prediction(member, 0.5, 0.0, 0.0, 0.0, "unparseable", text[:500])


def predict(idea: str, context: str, members: list[str] | None = None, model: str = "sonnet",
            parallel: int = 4) -> list[Prediction]:
    members = members or list(ROLES)

    def one(m: str) -> Prediction:
        system = CONTRACT + "\n# Your role\n\n" + ROLES[m] + "\nYou have no tools. Answer with the JSON object only."
        prompt = "# Context\n\n" + context + "\n\n# The idea\n\n" + idea + "\n\nReply with the JSON object only."
        return _parse(m, _ask(prompt, model, system=system))

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        return list(ex.map(one, members))


def aggregate(preds: list[Prediction]) -> dict:
    ps = [p.p_positive for p in preds]
    mean = sum(ps) / len(ps)
    spread = max(ps) - min(ps)
    var = sum((p - mean) ** 2 for p in ps) / len(ps)
    return {"p_mean": mean, "p_min": min(ps), "p_max": max(ps), "spread": spread, "std": var ** 0.5,
            "dpp_mean": sum(p.pred_dpp for p in preds) / len(preds),
            "dtime_mean": sum(p.pred_dtime for p in preds) / len(preds),
            "members": [p.__dict__ for p in preds]}
