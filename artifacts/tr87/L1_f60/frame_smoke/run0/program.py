# Mechanics: word-edit game. The cursor (player) sits under one editable glyph slot.
# ACTION3/ACTION4 move the cursor to the previous/next editable glyph x, wrapping around.
# ACTION1/2 cycle the selected glyph's letter, but glyphs carry no pixels -> no object change.
# The move counter's bar only changes pixels in the frame; its bbox stays fixed -> unchanged.
# Unconfirmed: ACTION5/6/7 effects (unobserved; treated as no-ops).
import copy


def _slots(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph" and "editable" in o.get("tags", [])})


def _move_cursor(cursor, slots, step):
    if cursor["x"] not in slots:
        return
    i = slots.index(cursor["x"])
    cursor["x"] = slots[(i + step) % len(slots)]


def transition_function(state, action, frame=None):
    out = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    slots = _slots(out)
    step = {3: -1, 4: 1}.get(aid)
    if step and slots:
        for o in out:
            if o.get("type") == "player":
                _move_cursor(o, slots, step)
    return out
