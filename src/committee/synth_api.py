"""Synthesis by a repair loop over any OpenAI-compatible chat model.

Round 1 sends the task, the transitions and the seed hypothesis. Each later
round sends the checker's report on the previous program. The loop stops at
exact replay or at the round budget. This is counterexample-guided inductive
synthesis with the exact-replay verifier as the oracle. The same loop runs
against a model served on Modal, a hosted API, or a local server.
"""

from __future__ import annotations

import os
import re
import time

from .env import contract, render, signature, stub
from .loader import Transition
from .synth import SynthResult
from .verify import train_report

SYSTEM = """\
You write Python world models for an unknown grid game. You get observed
transitions and must return the full source of `program.py`. Reply with one
```python code block and nothing else. The program must be self-contained,
standard library only, and define SIGNATURE.
"""


def api_contract(mode: str = "objects") -> str:
    return contract(mode).replace(
        "4. Run `python3 check.py` to see which transitions fail and how. Iterate until\n"
        "   it prints ALL PASS, or until you are convinced the remaining failures need\n"
        "   observations you do not have. Then stop.",
        "4. After each reply a checker runs your program on every transition and\n"
        "   reports the first failures. Repair the program and reply with the full\n"
        "   source again.",
    ).replace(
        "Files: `transitions.md` (readable diffs), `buffer.json` (full states),\n"
        "`program.py` (your code), `check.py` (the checker).\n",
        "",
    )


_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def load_env_file(path: str | None = None) -> None:
    """Load KEY=VALUE lines from the repo's .env into the environment, without overriding."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    paths = [Path(path)] if path else [root / ".env.committee", root / ".env"]
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def synthesize_any(train: list[Transition], seed: str | None, cfg: dict) -> SynthResult:
    """Dispatch on cfg["backend"]: "claude" runs the claude CLI agent, "api" runs the repair
    loop against an OpenAI-compatible endpoint (cfg: model, base_url, max_rounds, timeout_s)."""
    from .synth import synthesize

    backend = cfg.get("backend", "claude")
    mode = cfg.get("mode", "objects")
    if backend == "claude":
        return synthesize(train, seed, model=cfg.get("model", "opus"), max_turns=cfg.get("max_turns", 40),
                          timeout_s=cfg.get("timeout_s", 900), mode=mode)
    load_env_file()
    if backend == "devin":
        from .synth_devin import synthesize_devin

        return synthesize_devin(train, seed, timeout_s=cfg.get("timeout_s", 1800), mode=mode)
    return synthesize_api(train, seed, model=cfg["model"], base_url=cfg.get("base_url"),
                          max_rounds=cfg.get("max_rounds", 8), timeout_s=cfg.get("timeout_s", 900),
                          reasoning_effort=cfg.get("reasoning_effort"), mode=mode)


def extract_code(text: str) -> str | None:
    blocks = _CODE_RE.findall(text)
    if not blocks:
        return text.strip() if "def transition_function" in text else None
    return max(blocks, key=len).strip() + "\n"


def synthesize_api(train: list[Transition], seed: str | None = None, *, model: str,
                   base_url: str | None = None, api_key: str | None = None, max_rounds: int = 8,
                   timeout_s: float = 900, temperature: float = 0.7, max_tokens: int = 8000,
                   reasoning_effort: str | None = None, mode: str = "objects") -> SynthResult:
    from openai import OpenAI

    client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                    api_key=api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                    timeout=min(600.0, timeout_s))
    task = api_contract(mode) + ("\n# Hypothesis to build on\n\n" + seed.strip() + "\n" if seed else "")
    messages = [{"role": "system", "content": SYSTEM.replace("SIGNATURE", signature(mode))},
                {"role": "user", "content": task + "\n\n" + render(train, mode)}]
    t0 = time.time()
    meta: dict = {"model": model, "backend": "api", "base_url": base_url, "mode": mode, "rounds": 0,
                  "prompt_tokens": 0, "completion_tokens": 0, "round_log": []}
    source, report, passed = "", "", False
    best: tuple[int, str, str] = (-1, "", "")  # (train_pass, source, report): the loop may regress
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    for r in range(max_rounds):
        if time.time() - t0 > timeout_s:
            meta["timeout"] = True
            break
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                  max_tokens=max_tokens, extra_body=extra or None)
        except Exception as e:  # network or server error: keep what we have
            meta["round_log"].append({"round": r, "error": repr(e)[:300]})
            break
        text = resp.choices[0].message.content or ""
        if resp.usage:
            meta["prompt_tokens"] += resp.usage.prompt_tokens or 0
            meta["completion_tokens"] += resp.usage.completion_tokens or 0
        meta["rounds"] = r + 1
        code = extract_code(text)
        if code is None:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Reply with the full program in one ```python block."}]
            meta["round_log"].append({"round": r, "no_code": True})
            continue
        source = code
        passed, report = train_report(source, train, mode=mode)
        n_pass = int(report.split("/")[0]) if report and report[0].isdigit() else 0
        meta["round_log"].append({"round": r, "train_pass": n_pass, "bytes": len(source)})
        if n_pass > best[0]:
            best = (n_pass, source, report)
        if passed:
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Checker report:\n\n" + report[:6000]
                      + "\n\nRepair the program and reply with the full source."}]
    if not passed and best[0] >= 0:
        source, report = best[1], best[2]
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["passed"] = passed
    meta["best_train_pass"] = max(best[0], 0)
    return SynthResult(source=source or stub(mode), seed=seed, meta=meta, check_output=report[-2000:])
