"""Mechanics: the cursor (player) selects one editable glyph column.
Action 3 moves it to the previous editable glyph x, action 4 to the next; both wrap.
Actions 1 and 2 (likely cycle the selected glyph) have no visible effect in the extracted schema.
Hypothesis not confirmed: glyph letter changes, win check against reference words, counter decrement.
All other objects are static.
"""
import copy


def editable_columns(state):
    return sorted({o["x"] for o in state if "editable" in o.get("tags", [])})


def move_cursor(cursor, columns, step):
    if cursor["x"] not in columns:
        return
    i = columns.index(cursor["x"])
    cursor["x"] = columns[(i + step) % len(columns)]


def transition_function(state, action):
    after = copy.deepcopy(state)
    columns = editable_columns(after)
    step = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if step and columns:
        for o in after:
            if o["type"] == "player":
                move_cursor(o, columns, step)
    return after
