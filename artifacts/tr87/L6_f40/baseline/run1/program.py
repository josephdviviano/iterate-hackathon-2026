"""Mechanics: cursor (player) selects legend slots arranged in rows; its top is 4 above the row's legends.
Slots are legend columns; a legend within 7 cells of the previous one joins that slot (cursor skips it).
Action 4 moves to the next slot, wrapping to the first slot of the next row (last row wraps to first).
Action 3 moves to the previous slot; wrap to the previous row's last slot is assumed, not observed.
Actions 1, 2 (and others) change nothing observable; their editing effect is unconfirmed.
"""

ROW_OFFSET = 4
GROUP_GAP = 7


def slots(state):
    xs = sorted({o["x"] for o in state if o["type"] == "legend"})
    return [x for i, x in enumerate(xs) if i == 0 or x - xs[i - 1] > GROUP_GAP]


def rows(state):
    return sorted({o["y"] - ROW_OFFSET for o in state if o["type"] == "legend"})


def move_cursor(cursor, state, step):
    xs, ys = slots(state), rows(state)
    if cursor["x"] not in xs or cursor["y"] not in ys:
        return
    i = ys.index(cursor["y"]) * len(xs) + xs.index(cursor["x"])
    i = (i + step) % (len(xs) * len(ys))
    cursor["y"], cursor["x"] = ys[i // len(xs)], xs[i % len(xs)]


def transition_function(state, action):
    out = [dict(o) for o in state]
    step = {4: 1, 3: -1}.get(action if isinstance(action, int) else 0)
    if step:
        for o in out:
            if o["type"] == "player":
                move_cursor(o, state, step)
    return out
