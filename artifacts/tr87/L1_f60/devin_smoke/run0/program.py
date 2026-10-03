# Mechanics: a word-editing puzzle. The cursor (player) sits over one column of
# (reference target, editable glyph) pairs. Action 3 moves it one column left, action 4
# one column right, wrapping around the row of editable glyphs. Actions 1/2 (cycle the
# selected glyph) and 5/7 produce no visible field change (the extractor carries no pixels).
# Unconfirmed: a click (action 6) on a word column selects that column; the HUD counter never changes.
import copy


def editable_columns(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph" and "editable" in o.get("tags", [])})


def update_cursor(cursor, action, cols):
    if not cols:
        return
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx = action.get("x", -1)
            for c in cols:
                if c <= cx < c + cursor["w"]:
                    cursor["x"] = c
        return
    if action not in (3, 4) or cursor["x"] not in cols:
        return
    i = cols.index(cursor["x"])
    step = -1 if action == 3 else 1
    cursor["x"] = cols[(i + step) % len(cols)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    cols = editable_columns(new)
    for o in new:
        if o.get("type") == "player":
            update_cursor(o, action, cols)
    return new
