"""Synthesis by a Devin session (Cognition API v1).

Each committee member is one Devin session. The session gets the task, the
transitions and the checker as attachments, iterates in its own machine, and
returns the program through structured output. A blocked session (Devin asks
a question) is told to proceed with its best hypothesis. Needs DEVIN_API_KEY.
"""

from __future__ import annotations

import json
import os
import time

from .env import buffer_rows, check_script, contract, render, stub
from .loader import Transition
from .synth import SynthResult
from .verify import train_report

API = "https://api.devin.ai/v1"

SCHEMA = {
    "type": "object",
    "properties": {
        "program": {"type": "string", "description": "Full source of program.py"},
        "train_pass": {"type": "string", "description": "The checker's first line, e.g. 29/29"},
        "notes": {"type": "string", "description": "Mechanics implemented and open hypotheses, 5 lines"},
    },
    "required": ["program"],
}


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['DEVIN_API_KEY']}"}


def _retry(call, tries: int = 20, wait_s: float = 30):
    """Repeat a request over a dropped network. HTTP errors come back to the caller."""
    import httpx

    for i in range(tries):
        try:
            return call()
        except httpx.TransportError:
            if i == tries - 1:
                raise
            time.sleep(wait_s)


def _upload(client, name: str, text: str) -> str | None:
    try:
        r = _retry(lambda: client.post(f"{API}/attachments", headers=_headers(), files={"file": (name, text.encode())},
                                       timeout=120))
        r.raise_for_status()
        body = r.json()
        if isinstance(body, str):
            return body
        return body.get("url") if isinstance(body, dict) else None
    except Exception:
        return None


def devin_prompt(train: list[Transition], seed: str | None, attachments: dict[str, str | None],
                 mode: str = "objects") -> str:
    task = contract(mode).replace(
        "Files: `transitions.md` (readable diffs), `buffer.json` (full states),\n"
        "`program.py` (your code), `check.py` (the checker).\n", "")
    if seed:
        task += "\n# Hypothesis to build on\n\n" + seed.strip() + "\n"
    files = []
    inline = []
    for name, url in attachments.items():
        if url:
            files.append(f"- `{name}`: {url}")
        else:
            inline.append(name)
    setup = ("\n# Workspace\n\nDownload these files into one directory and work there:\n" + "\n".join(files) + "\n"
             if files else "")
    if "transitions.md" in inline:
        setup += "\n# transitions.md\n\n" + render(train, mode) + "\n"
    if "check.py" in inline:
        setup += "\n# check.py\n\n```python\n" + check_script(mode) + "\n```\n"
    if "buffer.json" in inline:
        setup += "\n# buffer.json\n\n```json\n" + json.dumps(buffer_rows(train, mode)) + "\n```\n"
    finish = ("\n# Finish\n\nWhen `python3 check.py` prints ALL PASS, or you cannot progress without new "
              "observations, return the structured output: the full program.py source, the checker's first "
              "line, and your notes. Do not ask questions; decide and proceed.\n")
    return task + setup + finish


def synthesize_devin(train: list[Transition], seed: str | None = None, *, timeout_s: float = 1800,
                     poll_s: float = 20, max_acu: int = 4, tags: list[str] | None = None,
                     mode: str = "objects") -> SynthResult:
    import httpx

    client = httpx.Client(timeout=60)
    attachments = {
        "transitions.md": _upload(client, "transitions.md", render(train, mode)),
        "buffer.json": _upload(client, "buffer.json", json.dumps(buffer_rows(train, mode))),
        "check.py": _upload(client, "check.py", check_script(mode)),
    }
    prompt = devin_prompt(train, seed, attachments, mode)
    t0 = time.time()
    meta: dict = {"backend": "devin", "mode": mode, "attachments_uploaded": sum(v is not None for v in attachments.values()),
                  "prompt_chars": len(prompt)}
    body = {"prompt": prompt, "structured_output_schema": SCHEMA, "max_acu_limit": max_acu, "unlisted": True,
            "tags": tags or ["committee"], "title": f"committee synth {time.strftime('%H:%M:%S')}"}
    r = _retry(lambda: client.post(f"{API}/sessions", headers=_headers(), json=body))
    if r.status_code != 200:
        raise RuntimeError(f"Devin create session {r.status_code}: {r.text[:500]} (prompt {len(prompt)} chars, "
                           f"attachments uploaded {meta['attachments_uploaded']}/3)")
    session = r.json()
    meta["session_id"], meta["session_url"] = session.get("session_id"), session.get("url")
    nudged, last_nudge = 0, 0.0
    status, out = None, {}
    nudge_text = ("Do not wait for me and stop refining. Return the structured output now: the full program.py "
                  "source, the checker's first line, and your notes. If the checker does not fully pass, return "
                  "your best program anyway.")
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        try:
            g = client.get(f"{API}/session/{meta['session_id']}", headers=_headers())
        except httpx.TransportError:  # the session keeps running on Devin while our network is down
            continue
        if g.status_code != 200:
            continue
        s = g.json()
        status = s.get("status_enum") or s.get("status")
        out = s.get("structured_output") or {}
        if status in ("finished", "expired") or (status == "blocked" and out.get("program")):
            break
        reported_pass = any("ALL PASS" in (m.get("message") or "") for m in (s.get("messages") or [])
                            if m.get("type") == "devin_message")
        # a blocked session is waiting for us; a working session that already reported ALL PASS is polishing
        if not out.get("program") and (status == "blocked" or reported_pass) and nudged < 4 \
                and time.time() - last_nudge > 150:
            try:
                client.post(f"{API}/session/{meta['session_id']}/message", headers=_headers(), json={"message": nudge_text})
            except httpx.TransportError:
                continue
            nudged, last_nudge = nudged + 1, time.time()
    meta.update({"status": status, "wall_s": round(time.time() - t0, 1), "nudges": nudged})
    return devin_result(out, meta, train, seed, mode)


def devin_result(out: dict, meta: dict, train: list[Transition], seed: str | None, mode: str) -> SynthResult:
    """The synthesis record from a session's structured output; the stub stands in for a missing program."""
    meta.update({"devin_train_pass": out.get("train_pass"), "notes": (out.get("notes") or "")[:1000]})
    source = out.get("program") or stub(mode)
    passed, report = train_report(source, train, mode=mode)
    meta["passed"] = passed
    return SynthResult(source=source, seed=seed, meta=meta, check_output=report[-2000:])
