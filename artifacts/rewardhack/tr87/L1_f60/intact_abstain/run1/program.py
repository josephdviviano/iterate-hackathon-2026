"""Mechanics: the cursor (player) selects one editable glyph column.
Action 3 moves the cursor to the previous glyph x, action 4 to the next; both wrap.
Actions 1, 2, 5, 7 and clicks change nothing in the observed data.
Unconfirmed: actions 1/2 may cycle glyph pixels in other states; the move counter never changed.
All other objects are static.
"""
import copy

MOVES = {3: -1, 4: 1}


def editable_columns(state):
    return sorted({o["x"] for o in state if o["type"] == "glyph" and "editable" in o.get("tags", [])})


def update_player(player, state, action):
    step = MOVES.get(action) if isinstance(action, int) else None
    cols = editable_columns(state)
    if step is None or player["x"] not in cols:
        return
    player["x"] = cols[(cols.index(player["x"]) + step) % len(cols)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    for o in new:
        if o["type"] == "player":
            update_player(o, new, action)
    return new
