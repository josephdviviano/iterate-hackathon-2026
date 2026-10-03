# Mechanics: word-editing puzzle. The cursor (player) sits over one editable glyph slot.
# ACTION3 moves the cursor to the previous editable glyph (by x), ACTION4 to the next; both wrap.
# ACTION1/ACTION2 (presumably cycling the selected glyph's letter) produce no observable change since
# objects carry no pixels; legends, targets, glyphs and the move counter are static.
# Unconfirmed: effect of ACTION1/2 on glyph appearance, ACTION5/6/7, and any win check vs ref words.
import copy


def _glyph_slots(state, cursor):
    """x positions of editable glyphs the cursor can select (glyph rows covered by the cursor)."""
    xs = sorted({o["x"] for o in state
                 if o.get("type") == "glyph"
                 and cursor["y"] <= o["y"] < cursor["y"] + cursor["h"]})
    if not xs:
        xs = sorted({o["x"] for o in state if o.get("type") == "glyph"})
    return xs


def _move_cursor(state, cursor, step):
    xs = _glyph_slots(state, cursor)
    if not xs:
        return
    if cursor["x"] in xs:
        i = xs.index(cursor["x"])
    else:
        i = min(range(len(xs)), key=lambda k: abs(xs[k] - cursor["x"]))
    cursor["x"] = xs[(i + step) % len(xs)]


def transition_function(state, action):
    out = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    for o in out:
        if o.get("type") == "player":
            if aid == 3:
                _move_cursor(out, o, -1)
            elif aid == 4:
                _move_cursor(out, o, +1)
    return out
