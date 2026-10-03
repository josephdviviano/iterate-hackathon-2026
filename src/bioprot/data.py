"""BioProt protocols as full-generation tasks.

Each protocol (O'Donoghue et al., 2023, arXiv:2310.10632) ships a title, a
human description, a GPT-4 description and pseudocode whose first part
defines the admissible pseudofunctions and whose second part calls them in
order. The task gives the model the title, one description and the function
definitions, and asks for the call sequence. The ground truth is the
expert-edited pseudocode where it exists, else the generated one.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "external" / "bioplanner" / "bioprot"

_DEF_LINE = re.compile(r"^def\s+[A-Za-z_]\w*\s*\(")
_CALL_LINE = re.compile(r"^\s*(?:[\w\s,\[\]]+?=\s*)?([A-Za-z_]\w*)\s*\(")
_KEYWORDS = {"print", "if", "for", "while", "with", "elif", "return", "def", "str", "int", "float", "len",
             "range", "list", "dict", "set", "tuple", "input", "open", "enumerate", "zip", "min", "max", "round"}


@dataclass(frozen=True)
class Protocol:
    id: str
    title: str
    description: str
    ai_description: str | None
    pseudocode: str
    definitions: tuple[str, ...]
    calls: tuple[str, ...]

    @property
    def function_names(self) -> tuple[str, ...]:
        return tuple(_def_name(d) for d in self.definitions)


def _def_name(block: str) -> str:
    return re.match(r"def\s+([A-Za-z_]\w*)", block).group(1)


def split_pseudocode(src: str) -> tuple[list[str], str]:
    """The `def` blocks (each with its indented body) and the remaining body text."""
    defs: list[str] = []
    body: list[str] = []
    block: list[str] | None = None
    for line in src.splitlines():
        if _DEF_LINE.match(line):
            if block:
                defs.append("\n".join(block).rstrip())
            block = [line]
        elif block is not None and (line.startswith((" ", "\t")) or not line.strip()):
            block.append(line)
        else:
            if block:
                defs.append("\n".join(block).rstrip())
                block = None
            body.append(line)
    if block:
        defs.append("\n".join(block).rstrip())
    return defs, "\n".join(body)


def _ast_calls(stmts) -> list[str]:
    out: list[str] = []
    for s in stmts:
        value = getattr(s, "value", None)
        if isinstance(s, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)) and isinstance(value, ast.Call):
            f = value.func
            out.append(f.id if isinstance(f, ast.Name) else ast.unparse(f))
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(s, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                out.extend(_ast_calls(inner))
    return out


def function_sequence(body: str) -> tuple[list[str], str]:
    """Statement-level call names in order. Python's parser when the text parses,
    else one call per line that starts with `name(` or `x = name(`. Returns the
    sequence and which parser produced it."""
    try:
        return _ast_calls(ast.parse(body).body), "ast"
    except (SyntaxError, ValueError):
        pass
    seq = []
    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _CALL_LINE.match(line)
        if m and m.group(1) not in _KEYWORDS:
            seq.append(m.group(1))
    return seq, "regex"


def strip_definitions(text: str) -> str:
    """A model reply without fences and without any function definitions it echoed."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    return split_pseudocode(text)[1]


def load_protocols(root: Path = ROOT) -> list[Protocol]:
    out = []
    for path in sorted(root.glob("*.json")):
        d = json.loads(path.read_text())
        src = d.get("edited_pseudocode") or d["generated_pseudocode"]
        defs, body = split_pseudocode(src)
        calls, _ = function_sequence(body)
        out.append(Protocol(id=str(d["id"]), title=d["title"].strip(), description=d["original description"].strip(),
                            ai_description=(d.get("ai_generated_description") or "").strip() or None,
                            pseudocode=src, definitions=tuple(defs), calls=tuple(calls)))
    return out


def shuffled_definitions(p: Protocol, seed: int) -> list[str]:
    """The definitions in a fixed random order per (protocol, seed). Unshuffled they
    appear in call order, which leaks the answer (BioPlanner, section 5)."""
    defs = list(p.definitions)
    random.Random(f"{p.id}:{seed}").shuffle(defs)
    return defs
