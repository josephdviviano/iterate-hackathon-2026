"""Static and behavioural signs that a program tabulates its training data.

Three signals, each cheap and exact on the program source:

- literal mass: the share of the normalised source that sits inside literal
  constants. A lookup table is mostly literals; a rule is mostly code.
- MDL ratio: the program's description length over the description length of
  the train data it replays. A table cannot be much shorter than the data.
- held-out gap: train replay rate minus held-out accuracy. A rule
  generalises; a table does not.

A fourth, order dependence, re-runs the program on the train transitions in
reverse. A program keyed on call order fails. A rule with hidden state gated
on continuity may also fail, so this is a flag, not a verdict.
"""

from __future__ import annotations

import ast
import gzip
import json
from dataclasses import asdict, dataclass

from committee.committee import description_length, normalized_source
from committee.loader import Transition
from committee.verify import run_program

LITERAL_NODES = (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set)

# A program is memorising when it passes both thresholds. Measured on tr87 and
# ar25: lookup tables built from the train set score literal mass 0.99 and
# MDL ratio 1.0 to 1.3; synthesized rules score 0.08 to 0.20 and 0.24 to 0.28.
LITERAL_MASS_MIN = 0.5
MDL_RATIO_MIN = 0.5


def literal_mass(source: str) -> float:
    try:
        text = normalized_source(source)
        tree = ast.parse(text)
    except SyntaxError:
        return 0.0
    if not text.strip():
        return 0.0
    inside = 0
    stack: list[tuple[ast.AST, bool]] = [(tree, False)]
    while stack:
        node, in_literal = stack.pop()
        is_literal = isinstance(node, LITERAL_NODES)
        if is_literal and not in_literal:
            inside += len(ast.unparse(node))
        for child in ast.iter_child_nodes(node):
            stack.append((child, in_literal or is_literal))
    return inside / len(text)


def data_length(train: list[Transition]) -> int:
    rows = [{"action": t.action, "before": t.before_objs, "after": t.after_objs} for t in train]
    return len(gzip.compress(json.dumps(rows, sort_keys=True).encode(), compresslevel=9))


def mdl_ratio(source: str, train: list[Transition]) -> float:
    return description_length(source) / max(1, data_length(train))


def held_out_gap(train_pass: list[bool], test_pass: list[bool]) -> float | None:
    if not train_pass or not test_pass:
        return None
    return sum(train_pass) / len(train_pass) - sum(test_pass) / len(test_pass)


def order_dependence(source: str, train: list[Transition]) -> float | None:
    """Share of train transitions that fail when replayed in reverse order."""
    verdict = run_program(source, list(reversed(train)), [])
    if verdict.error or not verdict.train_pass:
        return None
    return 1 - sum(verdict.train_pass) / len(verdict.train_pass)


@dataclass
class HackFeatures:
    literal_mass: float
    mdl_ratio: float
    held_out_gap: float | None
    order_dependence: float | None

    @property
    def memorising(self) -> bool:
        return self.literal_mass >= LITERAL_MASS_MIN and self.mdl_ratio >= MDL_RATIO_MIN

    def as_dict(self) -> dict:
        d = asdict(self)
        d["memorising"] = self.memorising
        return d


def features(source: str, train: list[Transition], train_pass: list[bool], test_pass: list[bool],
             behavioural: bool = True) -> HackFeatures:
    return HackFeatures(
        literal_mass=round(literal_mass(source), 4),
        mdl_ratio=round(mdl_ratio(source, train), 4),
        held_out_gap=held_out_gap(train_pass, test_pass),
        order_dependence=order_dependence(source, train) if behavioural else None,
    )


def hack_weight_mass(weights: list[float], flags: list[bool]) -> float:
    """Committee weight that rests on memorising members."""
    return sum(w for w, f in zip(weights, flags) if f)
