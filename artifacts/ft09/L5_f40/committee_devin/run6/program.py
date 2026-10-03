# Mechanics: lights-out board on an 8-cell grid; only ACTION6 clicks do anything.
# tile[editable]: clicking inside its bbox toggles color_14 <-> color_15.
# operator: clicking it toggles its own marker_14/15 and the tiles at (+-8,0)/(0,+-8);
#   glyph neighbours and empty cells are untouched. glyph: inert. Other actions: no-op.
# Naming: '<tile|glyph>_<y>_<x>' (operators share the glyph prefix); recomputed from position.
# Unconfirmed: effect of ACTION1-5/7 (never observed) and of clicking a glyph cell (assumed no-op).
import copy

SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}
STEP = 8


def name_of(o):
    prefix = "tile" if o["type"] == "tile" else "glyph"
    return "%s_%d_%d" % (prefix, o["y"], o["x"])


def toggle(o):
    o["tags"] = [SWAP.get(t, t) for t in o["tags"]]


def hit(o, x, y):
    return o["visible"] and o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def click_tile(state, tile):
    if "editable" in tile["tags"]:
        toggle(tile)


def click_operator(state, op):
    toggle(op)
    for dx, dy in ((STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)):
        for o in state:
            if o["type"] == "tile" and o["x"] == op["x"] + dx and o["y"] == op["y"] + dy:
                click_tile(state, o)


HANDLERS = {"tile": click_tile, "operator": click_operator}


def transition_function(state, action):
    state = copy.deepcopy(state)
    if isinstance(action, dict) and action.get("action_id") == 6:
        x, y = action["x"], action["y"]
        targets = sorted((o for o in state if hit(o, x, y)), key=lambda o: -o["layer"])
        if targets:
            handler = HANDLERS.get(targets[0]["type"])
            if handler:
                handler(state, targets[0])
    for o in state:
        o["name"] = name_of(o)
    return state
