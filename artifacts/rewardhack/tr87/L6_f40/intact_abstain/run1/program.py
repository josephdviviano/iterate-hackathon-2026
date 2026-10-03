"""Mechanics: legends form rows; cursor sits 4 cells above a selectable slot.
Slot = run of same-colour legends in one row separated by gaps < legend width.
Action 4 moves the cursor to the next slot (row-major, wraps to the first row);
action 3 moves to the previous slot. Other actions change nothing.
Unconfirmed: the slot-grouping rule, action 3 wrap, and what actions 1/2/5 edit.
"""
import copy


def _colour(o):
    return next(t for t in o["tags"] if t not in ("legend", "editable"))


def _slots(state):
    legends = sorted((o for o in state if o["type"] == "legend"),
                     key=lambda o: (o["y"], o["x"]))
    slots = []
    prev = None
    for o in legends:
        joins = (prev is not None and prev["y"] == o["y"]
                 and _colour(prev) == _colour(o)
                 and o["x"] - (prev["x"] + prev["w"]) < o["w"])
        if not joins:
            slots.append((o["x"], o["y"]))
        prev = o
    return slots


def transition_function(state, action):
    state = copy.deepcopy(state)
    if action not in (3, 4):
        return state
    cursor = next(o for o in state if o["type"] == "player")
    slots = _slots(state)
    pos = (cursor["x"], cursor["y"] + 4)
    if pos not in slots:
        return state
    i = slots.index(pos) + (1 if action == 4 else -1)
    cursor["x"], y = slots[i % len(slots)]
    cursor["y"] = y - 4
    return state
