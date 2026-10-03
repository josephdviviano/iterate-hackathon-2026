"""Mechanics: legends sit in rows; legends with a gap <= WORD_GAP join one word.
The cursor stops on the first legend of each word, top offset CURSOR_TOP_PAD.
Action 4 steps to the next word (wraps to next row, last row wraps to first);
action 3 steps back the same way. Actions 1 and 2 and others: no visible change.
Unconfirmed: transition 27 moves legend_cyan_0 x+1 with no observable cause.
"""

WORD_GAP = 2
CURSOR_TOP_PAD = 4


def _word_stops(state):
    legends = [o for o in state if o["type"] == "legend"]
    rows = {}
    for o in legends:
        rows.setdefault(o["y"], []).append(o)
    stops = []
    for y in sorted(rows):
        row = sorted(rows[y], key=lambda o: o["x"])
        prev = None
        for o in row:
            if prev is None or o["x"] - (prev["x"] + prev["w"]) > WORD_GAP:
                stops.append((o["x"], y - CURSOR_TOP_PAD))
            prev = o
    return stops


def _move_cursor(cursor, stops, step):
    pos = (cursor["x"], cursor["y"])
    if pos not in stops:
        return
    x, y = stops[(stops.index(pos) + step) % len(stops)]
    cursor["x"], cursor["y"] = x, y


def transition_function(state, action):
    out = [dict(o) for o in state]
    stops = _word_stops(out)
    step = {4: 1, 3: -1}.get(action) if isinstance(action, int) else None
    if step and stops:
        for o in out:
            if o["type"] == "player":
                _move_cursor(o, stops, step)
    return out
