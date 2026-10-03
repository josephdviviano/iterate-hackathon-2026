# Mechanics implemented:
# - ACTION6 click inside a tile bbox toggles that tile color_14<->color_15.
# - Click inside an operator bbox toggles its marker_14<->marker_15 and the
#   four orthogonal tiles at +/-8 px (missing neighbours are skipped).
# - Clicks on inert glyphs or empty cells, and actions 1-5/7, are no-ops.
# Unconfirmed: none; rule is stateless, hidden state was not needed.


def _flip(tags, a, b):
    return [b if t == a else a if t == b else t for t in tags]


def transition_function(state, action):
    out = [dict(o) for o in state]
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return out
    cx, cy = action.get("x"), action.get("y")
    hit = None
    for o in out:
        if o["x"] <= cx < o["x"] + o["w"] and o["y"] <= cy < o["y"] + o["h"]:
            hit = o
            break
    if hit is None:
        return out
    if hit["type"] == "tile":
        hit["tags"] = _flip(hit["tags"], "color_14", "color_15")
    elif hit["type"] == "operator":
        hit["tags"] = _flip(hit["tags"], "marker_14", "marker_15")
        for o in out:
            if o["type"] == "tile" and (
                (abs(o["x"] - hit["x"]) == 8 and o["y"] == hit["y"])
                or (abs(o["y"] - hit["y"]) == 8 and o["x"] == hit["x"])
            ):
                o["tags"] = _flip(o["tags"], "color_14", "color_15")
    return out
