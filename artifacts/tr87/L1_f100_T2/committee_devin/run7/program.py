# Mechanics: word-edit game. ACTION3/ACTION4 move the cursor (player) to the previous/next
# editable glyph slot (sorted glyph x), wrapping around; cursor y/size unchanged.
# ACTION1/ACTION2 presumably cycle the selected glyph's letter, but the schema has no pixels,
# so the edit is invisible -> no-op. Legends, reference targets and move_counter are static.
# Unconfirmed: ACTION5/6/7 (unobserved) assumed no-ops; no visible win/symmetry coupling.
import copy


def _slots(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph" and "editable" in o.get("tags", [])})


def _move_cursor(cursor, slots, step):
    if not slots:
        return
    if cursor["x"] in slots:
        i = (slots.index(cursor["x"]) + step) % len(slots)
    else:
        later = [s for s in slots if (s > cursor["x"]) == (step > 0)]
        i = slots.index(min(later, key=lambda s: abs(s - cursor["x"]))) if later else (0 if step > 0 else -1)
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
