# Mechanics: lights-out board of 6x6 cells on an 8-pixel grid; only ACTION6 clicks act.
# tile (editable): clicking it toggles its colour tag color_14 <-> color_15.
# operator: clicking it toggles its own marker_14 <-> marker_15 and toggles every
#   orthogonally adjacent tile (cell offset +-8); glyph/empty neighbours are untouched.
# glyph: inert; clicks on glyphs, empty space and non-click actions are no-ops. Stateless.
import copy

SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}


def toggle(obj):
    obj["tags"] = [SWAP.get(t, t) for t in obj["tags"]]


def hit(obj, x, y):
    return obj["x"] <= x < obj["x"] + obj["w"] and obj["y"] <= y < obj["y"] + obj["h"]


def is_neighbour(a, b, step=8):
    dx, dy = abs(a["x"] - b["x"]), abs(a["y"] - b["y"])
    return (dx, dy) in ((step, 0), (0, step))


def transition_function(state, action):
    out = copy.deepcopy(state)
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return out
    x, y = action["x"], action["y"]
    clicked = [o for o in out if o.get("visible", True) and hit(o, x, y)]
    if not clicked:
        return out
    c = max(clicked, key=lambda o: o.get("layer", 0))
    if c["type"] == "tile":
        toggle(c)
    elif c["type"] == "operator":
        toggle(c)
        for o in out:
            if o["type"] == "tile" and is_neighbour(o, c):
                toggle(o)
    return out
