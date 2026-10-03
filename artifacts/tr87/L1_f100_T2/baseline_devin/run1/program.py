# Mechanics: a cursor (player) selects one editable glyph column; ACTION3/ACTION4 move it to the
# previous/next editable glyph by x, wrapping around the ends (15 <-3- ... 43 -4-> 15).
# ACTION1/ACTION2 edit the selected glyph (cycle its letter), invisible in this pixel-free schema -> no change.
# Legends, reference words, counter are static. ACTION5/7/clicks: assumed no-op (unobserved).
# Unconfirmed: whether ACTION1/2 ever change glyph fields or the move counter; vertical cursor moves.
import copy


def _is_cursor(o):
    return o.get("type") == "player" or "cursor" in o.get("tags", [])


def _is_slot(o):
    return o.get("type") == "glyph" or "editable" in o.get("tags", [])


def _move_cursor(cursor, slots, step):
    xs = sorted({s["x"] for s in slots})
    if not xs:
        return
    if cursor["x"] in xs:
        i = (xs.index(cursor["x"]) + step) % len(xs)
    else:
        i = 0 if step > 0 else len(xs) - 1
    cursor["x"] = xs[i]


def transition_function(state, action):
    out = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    slots = [o for o in out if _is_slot(o)]
    for o in out:
        if _is_cursor(o):
            if aid == 3:
                _move_cursor(o, slots, -1)
            elif aid == 4:
                _move_cursor(o, slots, +1)
    return out
