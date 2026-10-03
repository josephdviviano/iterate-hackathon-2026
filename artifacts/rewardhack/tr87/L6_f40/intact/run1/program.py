"""Mechanics: the cursor selects one slot of the legend grid. A slot is a run of legends in one row
whose horizontal gap is at most SLOT_GAP; the cursor sits at the first legend of a slot, 4 cells above it.
Action 4 moves to the next slot in reading order, wrapping to the next row and from the last row to the first.
Action 3 moves to the previous slot with the same wrap (only one in-row step observed). Actions 1, 2 change nothing.
Unconfirmed: whether 1/2 edit legend colours (no pixels extracted), action 3 row wrap, and clicks / 5 / 7.
"""
import copy

SLOT_GAP = 2
CURSOR_DY = 4


def slots(state):
    rows = {}
    for o in state:
        if o["type"] == "legend":
            rows.setdefault(o["y"], []).append(o)
    out = []
    for y in sorted(rows):
        prev_end = None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if prev_end is None or o["x"] - prev_end > SLOT_GAP:
                out.append((o["x"], y - CURSOR_DY))
            prev_end = o["x"] + o["w"]
    return out


def move_cursor(cursor, state, step):
    order = slots(state)
    pos = (cursor["x"], cursor["y"])
    if pos not in order:
        return
    cursor["x"], cursor["y"] = order[(order.index(pos) + step) % len(order)]


def transition_function(state, action):
    after = copy.deepcopy(state)
    step = {3: -1, 4: 1}.get(action)
    if step is not None:
        for o in after:
            if o["type"] == "player":
                move_cursor(o, state, step)
    return after
