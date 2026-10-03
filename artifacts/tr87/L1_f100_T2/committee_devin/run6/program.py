# The cursor selects editable glyph slots; ACTION3/4 move left/right with wrap.
# Glyph slots are read from the state, not from fixed coordinates or names.
# Legend and reference objects retain all extracted fields; glyphs do too.
# The HUD counter has no value/pixels and remains unchanged in observations.
# ACTION1/2 are extractably unchanged; ACTION5/6/7 effects are unconfirmed.
from copy import deepcopy


def is_editable_glyph(obj):
    return obj.get("type") == "glyph" and "editable" in obj.get("tags", [])


def is_cursor_selector(obj):
    return "cursor" in obj.get("tags", []) and "selector" in obj.get("tags", [])


def is_navigation(action_id):
    return action_id in (3, 4)


def update_player(obj, action_id, slots):
    if is_cursor_selector(obj) and is_navigation(action_id) and obj.get("x") in slots:
        index = slots.index(obj["x"])
        direction = -1 if action_id == 3 else 1
        obj["x"] = slots[(index + direction) % len(slots)]
    return obj


def update_glyph(obj, action_id, slots):
    return obj


def update_legend(obj, action_id, slots):
    return obj


def update_target(obj, action_id, slots):
    return obj


def update_counter(obj, action_id, slots):
    return obj


def transition_function(state, action):
    action_id = action.get("action_id") if isinstance(action, dict) else action
    slots = sorted({obj["x"] for obj in state if is_editable_glyph(obj)})
    rules = {
        "player": update_player,
        "glyph": update_glyph,
        "legend": update_legend,
        "target": update_target,
        "counter": update_counter,
    }
    result = deepcopy(state)
    for obj in result:
        rule = rules.get(obj.get("type"))
        if rule is not None:
            rule(obj, action_id, slots)
    return result
