# Actions 3/4 move the cursor left/right, wrapping at the word ends.
# Selectable positions are inferred from editable glyphs, not fixed coordinates.
# Legends, reference targets, editable glyphs, and counters preserve all fields.
# Actions 1/2 have no extracted effect; letter identities/pixels are not exposed.
# Hidden edit history and actions 5/6/7 are unconfirmed; other actions are no-ops.
from copy import deepcopy


def _is_editable_glyph(obj):
    return obj.get("type") == "glyph" and "editable" in obj.get("tags", [])


def _cursor_on_word(obj, positions):
    return "cursor" in obj.get("tags", []) and obj.get("x") in positions


def _update_player(obj, action_id, positions):
    if action_id in (3, 4) and _cursor_on_word(obj, positions):
        index = positions.index(obj["x"])
        direction = -1 if action_id == 3 else 1
        obj["x"] = positions[(index + direction) % len(positions)]
    return obj


def _update_legend(obj, action_id, positions):
    return obj


def _update_target(obj, action_id, positions):
    return obj


def _update_glyph(obj, action_id, positions):
    return obj


def _update_counter(obj, action_id, positions):
    return obj


_RULES = {
    "player": _update_player,
    "legend": _update_legend,
    "target": _update_target,
    "glyph": _update_glyph,
    "counter": _update_counter,
}


def transition_function(state, action):
    result = deepcopy(state)
    action_id = action.get("action_id") if isinstance(action, dict) else action
    positions = sorted({obj["x"] for obj in state if _is_editable_glyph(obj)})
    for obj in result:
        rule = _RULES.get(obj.get("type"))
        if rule is not None:
            rule(obj, action_id, positions)
    return result
