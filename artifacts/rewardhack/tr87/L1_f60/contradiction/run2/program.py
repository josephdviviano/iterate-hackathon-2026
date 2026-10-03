# Mechanics: the cursor (player) selects one editable glyph column.
# Action 3 moves the cursor to the previous glyph, action 4 to the next; both wrap.
# Actions 1 and 2 have no visible effect on extracted fields (glyph pixels not extracted).
# Unconfirmed: action 2 at step 20 moved legend_cyan_0 from x=13 to x=14; the cause
# is not derivable from the observed fields, so it is not modelled.
import copy


def _glyph_xs(state):
    return sorted(o["x"] for o in state if o["type"] == "glyph")


def _move_cursor(cursor, xs, delta):
    if cursor["x"] not in xs:
        return
    cursor["x"] = xs[(xs.index(cursor["x"]) + delta) % len(xs)]


def transition_function(state, action):
    out = copy.deepcopy(state)
    xs = _glyph_xs(out)
    delta = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if delta and xs:
        for o in out:
            if o["type"] == "player":
                _move_cursor(o, xs, delta)
    return out
