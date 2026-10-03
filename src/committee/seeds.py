"""Seed hypotheses that steer each committee member toward a distinct explanation.

Unguided resampling of the same prompt tends to return the same program. Each
seed names one place the data is ambiguous, a mixed effect row, and one kind of
hidden condition that could resolve it. Generic seeds about simplicity, hidden
state, interactions and geometry fill the remaining slots.
"""

from __future__ import annotations

from itertools import cycle

from .loader import Transition
from .matrix import EffectMatrix

CONDITION_KINDS = [
    "a field of the object itself, such as its tags, pixels, visibility or layer",
    "the presence, type or relative position of a neighbouring object",
    "a global object such as a counter, HUD element or legend, read as state",
    "the exact click position relative to the object, when the action is a click",
    "state accumulated across earlier steps that the frame does not show",
]

GENERIC = [
    "Prefer the simplest stateless rule set. Treat every object type that never changes in "
    "the data as static scenery and model only the types that change.",
    "Assume some mechanic is not a function of the visible state, for example a turn order, "
    "a patrol route or a hidden counter. Carry hidden state across calls, gated on continuity.",
    "Express the mechanics as pairwise interactions between types with named guards, such as "
    "blocking, pushing, collecting or toggling, and derive single-type rules from them.",
    "Assume positions live on a lattice. Infer the step size and bounds from the data and write "
    "movement as lattice moves with explicit bound and obstacle checks.",
    "Start from the transitions where an action produced no change at all. Find what blocked the "
    "action in each case and write the blocking condition before the movement rule.",
    "Object names may encode position or identity. Work out the naming rule from the data and "
    "derive renaming on move, birth and removal from it, so names are never looked up.",
    "Treat counter, HUD, legend and indicator objects as state that other rules read, such as "
    "remaining moves, the selected item or goal progress, and model their updates exactly.",
    "Look for symmetry: mirrored or reflected objects, pairs that move together, or objects whose "
    "field equals a function of another object's field. Write those couplings as explicit rules.",
]


def make_seeds(train: list[Transition], k: int) -> list[str]:
    matrix = EffectMatrix.from_transitions(train)
    seeds: list[str] = []
    kinds = cycle(CONDITION_KINDS)
    for row in matrix.mixed_rows():
        if len(seeds) >= max(0, k - 2):
            break
        effects = ", ".join(f"{e} x{n}" for e, n in row.effects.most_common())
        seeds.append(
            f"Objects of type `{row.key.type}` under action {row.key.action} with touching "
            f"neighbours [{row.key.context}] show mixed outcomes: {effects}. Find the condition "
            f"in the before state that decides the outcome. Start from this kind of condition: "
            f"{next(kinds)}. Verify it against every transition before you keep it."
        )
    for g in GENERIC:
        if len(seeds) >= k:
            break
        seeds.append(g)
    while len(seeds) < k:
        seeds.append(GENERIC[len(seeds) % len(GENERIC)])
    return seeds[:k]
