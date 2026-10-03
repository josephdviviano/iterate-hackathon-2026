# Mechanics: lights-out board on an 8-cell grid (tiles/glyphs/operators are 6x6 at x,y).
# tile (editable): a click inside its bbox toggles tags color_14 <-> color_15.
# operator: a click inside toggles its marker_14 <-> marker_15 and toggles every tile
#   orthogonally adjacent at +-8 cells (glyph/operator neighbours are untouched).
# glyph: inert. Clicks on empty cells and non-click actions are no-ops. Stateless.
import copy

SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}


def toggle(obj):
    obj["tags"] = [SWAP.get(t, t) for t in obj["tags"]]


def hit(obj, x, y):
    return obj["x"] <= x < obj["x"] + obj["w"] and obj["y"] <= y < obj["y"] + obj["h"]


def is_operator_neighbour(op, obj):
    dx, dy = abs(obj["x"] - op["x"]), abs(obj["y"] - op["y"])
    return (dx, dy) in ((8, 0), (0, 8))


def transition_function(state, action):
    out = copy.deepcopy(state)
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return out
    x, y = action["x"], action["y"]
    target = next((o for o in out if o.get("visible", True) and hit(o, x, y)), None)
    if target is None:
        return out
    if target["type"] == "tile":
        toggle(target)
    elif target["type"] == "operator":
        toggle(target)
        for o in out:
            if o["type"] == "tile" and is_operator_neighbour(target, o):
                toggle(o)
    return out
