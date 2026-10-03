"""Mechanics: legends sit in rows; legends spaced exactly one word-pitch (7) apart form one group.
The cursor (y = row y - 4) stops at the first legend of each group. Action 4 moves to the next
stop, wrapping to the first stop of the next row (last row wraps to row 0); action 3 moves back.
Actions 1, 2 have no visible effect on extracted objects (maybe pixel-only edits, not extracted).
Unconfirmed: legend_cyan_0 shifts x+1 in transition 27 after an unobserved step; not modelled.
"""

WORD_PITCH = 7
CURSOR_DY = 4


def cursor_stops(state):
    rows = {}
    for o in state:
        if o["type"] == "legend":
            rows.setdefault(o["y"], []).append(o["x"])
    stops = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        for i, x in enumerate(xs):
            if i == 0 or x - xs[i - 1] != WORD_PITCH:
                stops.append((x, y - CURSOR_DY))
    return stops


def move_cursor(cursor, stops, step):
    pos = (cursor["x"], cursor["y"])
    if pos not in stops:
        return
    cursor["x"], cursor["y"] = stops[(stops.index(pos) + step) % len(stops)]


def transition_function(state, action):
    state = [dict(o) for o in state]
    stops = cursor_stops(state)
    step = {3: -1, 4: 1}.get(action)
    if step:
        for o in state:
            if o["type"] == "player":
                move_cursor(o, stops, step)
    return state
