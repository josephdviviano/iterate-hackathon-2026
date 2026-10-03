"""Mechanics: the cursor steps over legend groups, row by row. Legends in a row
whose gap is <= 2 cells form one group; the cursor stops at the group start.
Action 4 moves right, wrapping to the next row (last row wraps to first).
Action 3 moves left, wrapping to the previous row. Actions 1, 2, 5 change nothing.
Unconfirmed: what moved legend_cyan_0 by +1 x at step 164 (before state discontinuous).
"""

GROUP_GAP = 2


def legend_stops(state):
    rows = {}
    for o in state:
        if o["type"] == "legend":
            rows.setdefault(o["y"], []).append(o)
    stops = []
    for y in sorted(rows):
        prev_end = None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if prev_end is None or o["x"] - prev_end > GROUP_GAP:
                stops.append((o["x"], y))
            prev_end = o["x"] + o["w"]
    return stops


def cursor_stop_index(cursor, stops):
    for i, (x, y) in enumerate(stops):
        if cursor["x"] == x and cursor["y"] <= y < cursor["y"] + cursor["h"]:
            return i
    return None


def move_cursor(cursor, stops, step):
    i = cursor_stop_index(cursor, stops)
    if i is None:
        return
    x, y = stops[(i + step) % len(stops)]
    offset = cursor["y"] - stops[i][1]
    cursor["x"], cursor["y"] = x, y + offset


def transition_function(state, action):
    state = [dict(o) for o in state]
    if action in (3, 4):
        stops = legend_stops(state)
        for o in state:
            if o["type"] == "player":
                move_cursor(o, stops, 1 if action == 4 else -1)
    return state
