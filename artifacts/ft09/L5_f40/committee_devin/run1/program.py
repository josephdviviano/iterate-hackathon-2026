# Mechanics: lights-out board on an 8-cell grid; only ACTION6 clicks matter.
# tile: clicking a tile toggles its colour tag color_14 <-> color_15.
# operator: clicking it toggles its marker_14 <-> marker_15 and toggles every
#   tile orthogonally adjacent at +-8 (glyph neighbours are untouched).
# glyph / empty cells / other actions: no-op (static scenery). Unconfirmed: wins/level end.
import copy

SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}


def toggle(obj):
    obj["tags"] = [SWAP.get(t, t) for t in obj["tags"]]


def hit(obj, x, y):
    return obj["x"] <= x < obj["x"] + obj["w"] and obj["y"] <= y < obj["y"] + obj["h"]


def adjacent(a, b):
    dx, dy = abs(a["x"] - b["x"]), abs(a["y"] - b["y"])
    return (dx, dy) in ((8, 0), (0, 8))


def transition_function(state, action):
    out = copy.deepcopy(state)
    if not (isinstance(action, dict) and action.get("action_id") == 6):
        return out
    x, y = action["x"], action["y"]
    target = next((o for o in out if o.get("visible", True) and hit(o, x, y)
                   and o["type"] in ("tile", "operator")), None)
    if target is None:
        return out
    toggle(target)
    if target["type"] == "operator":
        for o in out:
            if o["type"] == "tile" and adjacent(o, target):
                toggle(o)
    return out
