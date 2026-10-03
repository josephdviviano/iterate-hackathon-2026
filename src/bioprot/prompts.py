"""Prompts for generation and for the two elicitation calls.

Elicitation is a separate call against the stored generation, so asking for
confidence cannot change the protocol that is judged.
"""

from __future__ import annotations

import re

from .data import Protocol

GENERATE_SYSTEM = """\
You write Python pseudocode for molecular biology protocols. You are given a
protocol title, a description and the admissible pseudofunctions. Write the
protocol steps as a sequence of calls to those functions, in execution order,
one call per line, with keyword arguments filled from the protocol.

Rules:
1. Call only the given functions. Do not define new functions and do not
   repeat the definitions.
2. Reply with one ```python block that contains only the calls, and nothing
   else.

Example reply for a protocol that pellets cells and washes them twice:
```python
centrifuge(sample="culture", speed="4000 x g", time="10 min")
discard_supernatant(sample="culture")
wash_pellet(sample="pellet", buffer="PBS", volume="1 mL")
wash_pellet(sample="pellet", buffer="PBS", volume="1 mL")
```
"""


def generation_messages(p: Protocol, definitions: list[str], description: str) -> list[dict]:
    user = (f"Title: {p.title}\n\nDescription:\n{description}\n\n"
            f"Pseudofunctions:\n" + "\n\n".join(definitions) + "\n\nProtocol steps:")
    return [{"role": "system", "content": GENERATE_SYSTEM}, {"role": "user", "content": user}]


CONFIDENCE_SYSTEM = """\
You review protocol plans before they run. You are given a protocol title, a
description, the admissible pseudofunctions and a proposed plan. Judge how
likely the plan is a correct and complete sequence of steps for this protocol:
the right functions, in the right order, with no step missing or extra.

Reply with exactly one line, either
CONFIDENCE: <integer 0-100>
or, when the title and description do not give enough information to write
this protocol safely,
ABSTAIN: <one sentence>
"""


CRITIQUE_SYSTEM = """\
You are the lab lead. A protocol plan is about to be executed on a liquid
handling robot. You are given the protocol title, a description, the
admissible pseudofunctions and the plan. Decide whether to let it run.

Reply with exactly one word on the first line, PASS or FAIL, then one
sentence of reason on the second line.
"""


def _context(p: Protocol, definitions: list[str], description: str, plan: str) -> str:
    return (f"Title: {p.title}\n\nDescription:\n{description}\n\n"
            f"Pseudofunctions:\n" + "\n\n".join(definitions) + f"\n\nProposed plan:\n```python\n{plan.strip()}\n```")


def confidence_messages(p: Protocol, definitions: list[str], description: str, plan: str) -> list[dict]:
    return [{"role": "system", "content": CONFIDENCE_SYSTEM},
            {"role": "user", "content": _context(p, definitions, description, plan) + "\n\nYour line:"}]


def critique_messages(p: Protocol, definitions: list[str], description: str, plan: str) -> list[dict]:
    return [{"role": "system", "content": CRITIQUE_SYSTEM},
            {"role": "user", "content": _context(p, definitions, description, plan) + "\n\nYour decision:"}]


_CONF = re.compile(r"CONFIDENCE\s*[:=]\s*(\d{1,3})", re.I)
_ABSTAIN = re.compile(r"^\s*ABSTAIN\b", re.I | re.M)
_DECLINE = re.compile(r"\b(I can(?:no|')t|I cannot|I'm unable|I am unable|not able to (?:help|assist|provide))\b", re.I)


def parse_confidence(reply: str) -> dict:
    """confidence in [0, 1] or None; abstained (the offered channel); declined (a
    refusal that used neither channel). The two events are stored apart."""
    m = _CONF.search(reply)
    if m:
        return {"confidence": min(int(m.group(1)), 100) / 100, "abstained": False, "declined": False}
    if _ABSTAIN.search(reply):
        return {"confidence": None, "abstained": True, "declined": False}
    return {"confidence": None, "abstained": False, "declined": bool(_DECLINE.search(reply))}


def parse_verdict(reply: str) -> str | None:
    first = reply.strip().split("\n", 1)[0].strip().strip("*`.:").upper()
    if first.startswith("PASS"):
        return "PASS"
    if first.startswith("FAIL"):
        return "FAIL"
    m = re.search(r"\b(PASS|FAIL)\b", reply.upper())
    return m.group(1) if m else None
