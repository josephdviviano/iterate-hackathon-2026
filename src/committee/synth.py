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

from .env import CHECK_SCRIPT, CONTRACT, STUB, buffer_rows, check_script, contract, render, render_transitions, stub
from .loader import Transition

__all__ = ["CHECK_SCRIPT", "CONTRACT", "STUB", "SynthResult", "render_transitions", "synthesize"]

@dataclass
class SynthResult:
    source: str
    seed: str | None
    meta: dict = field(default_factory=dict)
    check_output: str = ""


def write_workspace(ws: Path, train: list[Transition], seed: str | None, mode: str = "objects") -> None:
    """The agent's files: task, readable transitions, the buffer the checker replays, the checker
    and a stub, all in the form the mode fixes. Test transitions never enter the workspace."""
    task = contract(mode)
    if seed:
        task += "\n# Hypothesis to build on\n\n" + seed.strip() + "\n"
    (ws / "TASK.md").write_text(task)
    (ws / "transitions.md").write_text(render(train, mode))
    (ws / "buffer.json").write_text(json.dumps(buffer_rows(train, mode)))
    (ws / "check.py").write_text(check_script(mode))
    (ws / "program.py").write_text(stub(mode))


def synthesize(train: list[Transition], seed: str | None = None, *, model: str = "opus",
               max_turns: int = 40, timeout_s: float = 900, keep_dir: Path | None = None,
               mode: str = "objects") -> SynthResult:
    ws = Path(tempfile.mkdtemp(prefix="synth_"))
    write_workspace(ws, train, seed, mode)
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
    meta: dict = {"model": model, "max_turns": max_turns, "mode": mode}
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
