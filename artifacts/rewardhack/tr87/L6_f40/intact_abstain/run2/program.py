"""Mechanics: the cursor selects one stop among legend rows. A stop is a run of
legends in a row whose gaps are <= 2 cells (letters of one word); cursor sits at
the stop's first legend x, 4 cells above it. Action 4 moves to the next stop in
reading order, wrapping to the first stop. Action 3 moves to the previous stop
(wrap at row start unconfirmed). Actions 1, 2, 5, 7 and clicks change nothing.
"""

import copy

WORD_GAP = 2
CURSOR_ABOVE = 4


def legend_stops(state):
    legends = [o for o in state if o["type"] == "legend"]
    rows = {}
    for o in legends:
        rows.setdefault(o["y"], []).append(o)
    stops = []
    for y in sorted(rows):
        prev_end = None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if prev_end is None or o["x"] - prev_end > WORD_GAP:
                stops.append((o["x"], y - CURSOR_ABOVE))
            prev_end = o["x"] + o["w"]
    return stops


def move_cursor(cursor, stops, step):
    pos = (cursor["x"], cursor["y"])
    if pos not in stops:
        return
    x, y = stops[(stops.index(pos) + step) % len(stops)]
    cursor["x"], cursor["y"] = x, y


def transition_function(state, action):
    state = copy.deepcopy(state)
    step = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if step is None:
        return state
    stops = legend_stops(state)
    for o in state:
        if o["type"] == "player":
            move_cursor(o, stops, step)
    return state
