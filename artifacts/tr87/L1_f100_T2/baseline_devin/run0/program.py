# Mechanics: word-editing puzzle. Cursor (player) sits over one editable glyph slot.
# ACTION3/ACTION4 move the cursor to the previous/next editable glyph (by x), wrapping around.
# ACTION1/ACTION2 cycle the selected glyph's letter; the extractor carries no pixels, so no visible change.
# Legends, reference words, glyph boxes and the HUD counter (fixed width) are static.
# Unconfirmed: ACTION5/6/7 effects (unobserved, treated as no-ops); win/level change when words match.
import copy


def _slots(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph"})


def _update_cursor(cur, slots, action):
    if not slots or action not in (3, 4):
        return
    if cur["x"] in slots:
        i = slots.index(cur["x"])
    else:
        i = min(range(len(slots)), key=lambda k: abs(slots[k] - cur["x"]))
    i = (i - 1) % len(slots) if action == 3 else (i + 1) % len(slots)
    cur["x"] = slots[i]


def transition_function(state, action):
    out = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    slots = _slots(out)
    for o in out:
        if o.get("type") == "player":
            _update_cursor(o, slots, aid)
    return out
