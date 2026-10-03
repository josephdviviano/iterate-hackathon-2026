# Mechanics: ACTION6 click hit-tests objects by bbox [x,x+w) x [y,y+h).
# tile: clicking an 'editable' tile toggles its color tag color_14 <-> color_15.
# operator: clicking toggles its own marker_14 <-> marker_15 and toggles every tile whose
#   bbox origin is one 8-cell grid step away orthogonally (glyphs/operators there untouched).
# glyph: inert; clicks on empty cells and non-click actions are no-ops. Unconfirmed: whether
#   operators also flip neighbouring operators/glyphs, and what actions 1-5/7 do (unobserved).
import copy

STEP = 8
SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}


def toggle(obj):
    obj["tags"] = [SWAP.get(t, t) for t in obj["tags"]]


def hit(obj, x, y):
    return obj["visible"] and obj["x"] <= x < obj["x"] + obj["w"] and obj["y"] <= y < obj["y"] + obj["h"]


def is_neighbour(a, b):
    dx, dy = abs(a["x"] - b["x"]), abs(a["y"] - b["y"])
    return (dx, dy) in ((STEP, 0), (0, STEP))


def click_tile(state, tile):
    toggle(tile)


def click_operator(state, op):
    toggle(op)
    for o in state:
        if o["type"] == "tile" and is_neighbour(o, op):
            toggle(o)


HANDLERS = {"tile": click_tile, "operator": click_operator}


def transition_function(state, action):
    state = copy.deepcopy(state)
    if not (isinstance(action, dict) and action.get("action_id") == 6):
        return state
    x, y = action["x"], action["y"]
    hits = [o for o in state if hit(o, x, y)]
    if not hits:
        return state
    target = max(hits, key=lambda o: o.get("layer", 0))
    handler = HANDLERS.get(target["type"])
    if handler:
        handler(state, target)
    return state
