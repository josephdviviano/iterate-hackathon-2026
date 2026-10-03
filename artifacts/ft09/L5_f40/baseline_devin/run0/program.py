# Mechanics: lights-out style grid of 6x6 cells spaced 8 apart. ACTION6 click hits the object whose bbox contains (x,y).
# - tile (editable): click toggles its color tag color_14 <-> color_15.
# - operator: click toggles its own marker_14 <-> marker_15 and toggles the 4 orthogonally adjacent editable tiles
#   (neighbors at offset w+2 / h+2); adjacent glyphs/operators are not affected. glyph: click is a no-op.
# - Non-click actions and clicks on empty space: no change. Hypothesis unconfirmed: other colors/markers beyond 14/15 cycle similarly.
import copy

SWAP = {"14": "15", "15": "14"}


def _toggle(obj, prefix):
    tags = []
    for t in obj["tags"]:
        if t.startswith(prefix):
            v = t[len(prefix):]
            t = prefix + SWAP.get(v, v)
        tags.append(t)
    obj["tags"] = tags


def _hit(state, x, y):
    for o in state:
        if o.get("visible", True) and o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]:
            return o
    return None


def _is_tile(o):
    return o["type"] == "tile" and "editable" in o["tags"]


def _click_tile(state, o):
    _toggle(o, "color_")


def _click_operator(state, o):
    _toggle(o, "marker_")
    dx, dy = o["w"] + 2, o["h"] + 2
    for nx, ny in ((o["x"] - dx, o["y"]), (o["x"] + dx, o["y"]), (o["x"], o["y"] - dy), (o["x"], o["y"] + dy)):
        for n in state:
            if _is_tile(n) and n["x"] == nx and n["y"] == ny:
                _toggle(n, "color_")


def transition_function(state, action):
    state = copy.deepcopy(state)
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return state
    o = _hit(state, action["x"], action["y"])
    if o is None:
        return state
    if _is_tile(o):
        _click_tile(state, o)
    elif o["type"] == "operator":
        _click_operator(state, o)
    return state
