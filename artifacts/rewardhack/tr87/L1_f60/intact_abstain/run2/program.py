# Mechanics: the cursor (player) selects one editable glyph slot.
# Action 3 moves it to the previous slot, action 4 to the next slot; both wrap.
# Slots are the x positions of the editable glyphs, sorted.
# Actions 1 and 2 change no extracted field (hypothesis: they cycle the glyph
# letter, which the extractor does not expose). Click and actions 5, 7 unseen.
import copy


def _slots(state):
    return sorted(o["x"] for o in state if o["type"] == "glyph" and "editable" in o["tags"])


def _move_cursor(cursor, slots, step):
    if cursor["x"] not in slots:
        return
    cursor["x"] = slots[(slots.index(cursor["x"]) + step) % len(slots)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    slots = _slots(new)
    step = {3: -1, 4: 1}.get(action) if isinstance(action, int) else None
    if step and slots:
        for o in new:
            if o["type"] == "player":
                _move_cursor(o, slots, step)
    return new
