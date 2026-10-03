# Mechanics: word-edit game. Only the cursor (type 'player') ever changes.
# ACTION3/ACTION4 move the cursor to the previous/next editable glyph x, wrapping around.
# ACTION1/2 (glyph edits) have no visible effect: the schema has no pixels, so glyphs stay unchanged.
# Legends, reference words, glyphs and the move counter are static scenery; ACTION5/6/7 are no-ops.
# Unconfirmed: cursor movement when glyphs are not on one row; counter never changes in the data.
import copy


def _glyph_xs(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph"})


def _move_cursor(cursor, xs, step):
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
    step = {3: -1, 4: 1}.get(aid)
    if step is not None:
        xs = _glyph_xs(out)
        for o in out:
            if o.get("type") == "player":
                _move_cursor(o, xs, step)
    return out
