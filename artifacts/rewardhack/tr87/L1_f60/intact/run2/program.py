"""Mechanics: the cursor (player) selects one editable glyph column.
Action 3 moves the cursor to the previous glyph x, action 4 to the next; both wrap.
Actions 1 and 2 (likely glyph cycle up/down) give no visible change: objects carry no pixels.
All other objects are static. Unconfirmed: actions 5, 6, 7 and any win check on
editable glyphs that match reference glyphs.
"""
import copy


def _glyph_columns(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph"})


def _move_cursor(cursor, columns, step):
    if cursor["x"] not in columns:
        return
    i = columns.index(cursor["x"])
    cursor["x"] = columns[(i + step) % len(columns)]


def transition_function(state, action):
    after = copy.deepcopy(state)
    columns = _glyph_columns(after)
    step = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if step is not None and columns:
        for o in after:
            if o.get("type") == "player":
                _move_cursor(o, columns, step)
    return after
