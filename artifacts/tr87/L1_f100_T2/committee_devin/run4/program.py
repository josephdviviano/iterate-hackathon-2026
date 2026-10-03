# Mechanics: word-edit game. Cursor (player) sits over one editable glyph column.
# ACTION3/ACTION4 move the cursor to the previous/next editable glyph x (wraparound).
# Blocking: ACTION1/2 (glyph edit) and ACTION5/6/7 change no visible field (schema has
# no pixels), so they are no-ops; legends, targets, glyphs and move_counter are static.
# Unconfirmed: ACTION1/2 likely cycle the selected glyph's letter, invisible here.
import copy


def _slots(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph" and "editable" in o.get("tags", [])})


def _move_cursor(cursor, slots, step):
    if not slots:
        return
    if cursor["x"] in slots:
        i = (slots.index(cursor["x"]) + step) % len(slots)
    else:
        i = 0 if step > 0 else len(slots) - 1
    cursor["x"] = slots[i]


def transition_function(state, action):
    out = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    step = {3: -1, 4: 1}.get(aid)
    if step is None:
        return out
    slots = _slots(out)
    for o in out:
        if o.get("type") == "player":
            _move_cursor(o, slots, step)
    return out
