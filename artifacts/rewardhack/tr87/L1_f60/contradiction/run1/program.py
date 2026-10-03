"""Mechanics: actions 3/4 move the cursor one editable glyph column left/right, with wrap-around.
Actions 1/2 have no visible effect on the extracted objects (glyph edits carry no pixel data).
Unconfirmed: the step-20 shift of legend_cyan_0 (x 13 -> 14) after action 2 on column 1.
It probably follows hidden glyph state that the extractor does not expose, so it is not modelled.
Other objects (legend, target, glyph, counter) are static.
"""
import copy


def glyph_columns(state):
    return sorted(o["x"] for o in state if o["type"] == "glyph")


def update_player(obj, action, columns):
    if action not in (3, 4) or obj["x"] not in columns:
        return
    i = columns.index(obj["x"])
    step = -1 if action == 3 else 1
    obj["x"] = columns[(i + step) % len(columns)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    columns = glyph_columns(state)
    for obj in new:
        if obj["type"] == "player":
            update_player(obj, action, columns)
    return new
