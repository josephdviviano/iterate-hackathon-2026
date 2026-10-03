# Mechanics: word-edit game. Cursor (player) sits on a 1-D lattice of editable glyph columns
#   (step = glyph spacing, 7 here; bounds = first/last glyph). ACTION3 = previous slot, ACTION4 =
#   next slot, both wrapping around at the bounds. Legends, targets, glyphs and counter are static.
# Hypothesis (unconfirmed): ACTION1/2 edit the glyph under the cursor, but the schema has no pixels,
#   so edits are invisible -> no-op; the move counter never changes in the data. Stateless.
import copy


def _slots(state):
    """Lattice of cursor positions: x of every editable glyph, sorted."""
    return sorted({o["x"] for o in state if o.get("type") == "glyph" and "editable" in o.get("tags", [])})


def _move_cursor(cursor, slots, delta):
    if not slots:
        return
    if cursor["x"] in slots:
        i = slots.index(cursor["x"])
    else:
        i = min(range(len(slots)), key=lambda k: abs(slots[k] - cursor["x"]))
    cursor["x"] = slots[(i + delta) % len(slots)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    delta = {3: -1, 4: 1}.get(aid)
    if delta is not None:
        slots = _slots(new)
        for o in new:
            if o.get("type") == "player":
                _move_cursor(o, slots, delta)
    return new
