"""Mechanics: the cursor (player) selects one editable glyph column.
Action 3 moves the cursor one column left, action 4 one column right; both wrap.
Actions 1 and 2 have no visible effect on object geometry (glyph pixels are not extracted).
Unconfirmed: action 2 at step 20 moved legend_cyan_0 by +1 in x. One observation does not
identify the rule (likely a bbox change from a hidden pixel edit), so it is not modelled.
"""

import copy


def _columns(state):
    return sorted(o["x"] for o in state if o["type"] == "glyph")


def _move_cursor(cursor, columns, step):
    if cursor["x"] not in columns:
        return
    i = (columns.index(cursor["x"]) + step) % len(columns)
    cursor["x"] = columns[i]


def transition_function(state, action):
    out = copy.deepcopy(state)
    columns = _columns(out)
    for o in out:
        if o["type"] == "player" and columns:
            if action == 3:
                _move_cursor(o, columns, -1)
            elif action == 4:
                _move_cursor(o, columns, 1)
    return out
