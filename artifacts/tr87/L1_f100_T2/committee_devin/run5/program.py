# Mechanics: word-edit game. ACTION3/ACTION4 move the cursor (player) to the previous/next
# editable glyph slot (sorted glyph x), wrapping around; cursor y/w/h unchanged.
# ACTION1/2 (glyph edits) and others have no visible effect: the schema has no pixels,
# so glyph identity changes are unobservable -> no-op. Counter, legends, targets static.
# Hypothesis unconfirmed: ACTION1/2 cycle the selected glyph's letter; names stay type_i by x.
import copy


def _slots(state):
    return sorted(o["x"] for o in state if o.get("type") == "glyph")


def _move_cursor(cursor, slots, step):
    if not slots:
        return
    if cursor["x"] in slots:
        i = (slots.index(cursor["x"]) + step) % len(slots)
    else:
        i = 0 if step > 0 else len(slots) - 1
    cursor["x"] = slots[i]


def transition_function(state, action):
    new = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    step = {3: -1, 4: 1}.get(aid)
    if step is not None:
        slots = _slots(new)
        for o in new:
            if o.get("type") == "player":
                _move_cursor(o, slots, step)
    return new
