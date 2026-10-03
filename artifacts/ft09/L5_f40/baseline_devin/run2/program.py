# Mechanics: ACTION6 click at (x,y) hits the object whose bbox contains it.
#  - editable tile: toggles its colour tag color_14 <-> color_15 (in place in tags).
#  - operator: toggles its own marker_14 <-> marker_15 and toggles every editable
#    tile whose top-left is one grid step (8 cells) away orthogonally; glyphs untouched.
#  - glyphs / empty cells / other actions: no-op. Unconfirmed: non-click actions, win state.
import copy

STEP = 8
SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}


def _toggle(obj):
    obj["tags"] = [SWAP.get(t, t) for t in obj["tags"]]


def _contains(o, x, y):
    return o.get("visible", True) and o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def _is_tile(o):
    return o["type"] == "tile" and "editable" in o["tags"]


def _operator_neighbours(state, op):
    offs = {(STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)}
    return [o for o in state if _is_tile(o) and (o["x"] - op["x"], o["y"] - op["y"]) in offs]


def transition_function(state, action):
    new = copy.deepcopy(state)
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return new
    x, y = action["x"], action["y"]
    hits = sorted((o for o in new if _contains(o, x, y)), key=lambda o: -o.get("layer", 0))
    if not hits:
        return new
    target = hits[0]
    if _is_tile(target):
        _toggle(target)
    elif target["type"] == "operator":
        _toggle(target)
        for t in _operator_neighbours(new, target):
            _toggle(t)
    return new
