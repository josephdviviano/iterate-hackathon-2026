"""Mechanics: the cursor (player) selects one editable glyph column.
Action 3 moves it to the previous glyph column, action 4 to the next; both wrap.
Actions 1 and 2 change nothing in the observations (unconfirmed: they may cycle
the selected glyph under conditions not observed). Legends, targets, glyphs and
the move counter stay unchanged.
"""
import copy


def glyph_columns(state):
    return sorted(o["x"] for o in state if o["type"] == "glyph")


def move_cursor(cursor, columns, step):
    if cursor["x"] not in columns:
        return
    i = columns.index(cursor["x"])
    cursor["x"] = columns[(i + step) % len(columns)]


def transition_function(state, action):
    after = copy.deepcopy(state)
    columns = glyph_columns(after)
    step = {3: -1, 4: 1}.get(action, 0) if isinstance(action, int) else 0
    for o in after:
        if o["type"] == "player" and step and columns:
            move_cursor(o, columns, step)
    return after
