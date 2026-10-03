"""Mechanics: the cursor selects one editable glyph column.
Action 3 moves the cursor to the previous glyph column, action 4 to the next; both wrap around.
Actions 1, 2, 5, 7 and clicks cause no observed change.
Hypothesis not confirmed: actions 1/2 may change the selected glyph, but no glyph change was seen.
The move counter did not change in any observation, so it is kept as is.
"""
import copy


def glyph_columns(state):
    return sorted({o["x"] for o in state if o["type"] == "glyph"})


def update_cursor(cursor, columns, action):
    if cursor["x"] not in columns:
        return
    i = columns.index(cursor["x"])
    if action == 3:
        cursor["x"] = columns[(i - 1) % len(columns)]
    elif action == 4:
        cursor["x"] = columns[(i + 1) % len(columns)]


def transition_function(state, action):
    after = copy.deepcopy(state)
    columns = glyph_columns(after)
    for o in after:
        if o["type"] == "player":
            update_cursor(o, columns, action)
    return after
