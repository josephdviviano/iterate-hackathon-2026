"""Mechanics: the cursor steps through legend slots in reading order.
Legends on one row that are separated by a gap of 2 cells or less form one slot;
the cursor sits on the first legend of a slot, aligned to its x, 4 cells above.
Action 4 = next slot, action 3 = previous slot; both wrap across rows and the grid.
Unconfirmed: actions 1, 2 (no visible effect, maybe hidden edit), 5, 6, 7 assumed no-op.
"""

import copy

SLOT_GAP = 2
CURSOR_OFFSET_Y = 4


def legend_slots(state):
    legends = [o for o in state if o["type"] == "legend"]
    rows = {}
    for o in legends:
        rows.setdefault(o["y"], []).append(o)
    slots = []
    for y in sorted(rows):
        prev_end = None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if prev_end is None or o["x"] - prev_end > SLOT_GAP:
                slots.append((o["x"], y - CURSOR_OFFSET_Y))
            prev_end = o["x"] + o["w"]
    return slots


def move_cursor(cursor, slots, step):
    pos = (cursor["x"], cursor["y"])
    if pos not in slots:
        return
    x, y = slots[(slots.index(pos) + step) % len(slots)]
    cursor["x"], cursor["y"] = x, y


def transition_function(state, action):
    state = copy.deepcopy(state)
    step = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if step is None:
        return state
    slots = legend_slots(state)
    for o in state:
        if o["type"] == "player":
            move_cursor(o, slots, step)
    return state
