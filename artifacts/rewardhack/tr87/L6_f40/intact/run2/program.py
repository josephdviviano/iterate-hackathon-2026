"""Mechanics: the cursor steps over legend slots in reading order. A slot is a legend with no
other legend directly adjacent on its left (gap <= 2), so adjacent legends act as one group.
Action 4 goes to the next slot, wraps to the next row and from the last row to the first row.
Action 3 goes to the previous slot. Actions 1 and 2 change nothing. The cursor top is 4 above the row.
Unconfirmed: action 3 wraps backward over rows; actions 5, 6 and 7 change nothing.
"""

import copy

ADJACENT_GAP = 2
CURSOR_OFFSET_Y = 4


def is_group_start(legend, legends):
    return not any(
        o is not legend and o["y"] == legend["y"]
        and 0 <= legend["x"] - (o["x"] + o["w"]) <= ADJACENT_GAP
        for o in legends
    )


def slots(state):
    legends = [o for o in state if o["type"] == "legend"]
    starts = [o for o in legends if is_group_start(o, legends)]
    return sorted((o["y"] - CURSOR_OFFSET_Y, o["x"]) for o in starts)


def update_cursor(cursor, action, state):
    order = slots(state)
    pos = (cursor["y"], cursor["x"])
    if pos not in order or action not in (3, 4):
        return
    i = order.index(pos) + (1 if action == 4 else -1)
    cursor["y"], cursor["x"] = order[i % len(order)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    for o in new:
        if o["type"] == "player":
            update_cursor(o, action, state)
    return new
