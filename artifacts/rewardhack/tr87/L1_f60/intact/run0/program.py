"""Mechanics: the cursor selects one editable glyph column. Action 3 moves it to
the previous glyph and action 4 to the next glyph, with wrap-around at the ends.
Actions 1 and 2 give no visible change (unconfirmed: they may cycle the glyph's
letter, but the extractor carries no pixels to show it). Action 5, 6, 7 not
observed; they are treated as no-ops. The move counter did not change.
"""
import copy


def _glyph_columns(state):
    return sorted(o["x"] for o in state if o.get("type") == "glyph")


def _move_cursor(cursor, columns, step):
    if not columns:
        return
    if cursor["x"] in columns:
        i = columns.index(cursor["x"])
    else:
        i = 0
    cursor["x"] = columns[(i + step) % len(columns)]


def transition_function(state, action):
    after = copy.deepcopy(state)
    step = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if step is not None:
        columns = _glyph_columns(after)
        for o in after:
            if o.get("type") == "player":
                _move_cursor(o, columns, step)
    return after
