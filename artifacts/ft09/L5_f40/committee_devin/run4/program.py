# Mechanics: lights-out board on a lattice (step 8, cells 6x6, origin offset 6/4); only ACTION6 clicks act.
# tile (editable): clicking inside its bbox toggles color_14 <-> color_15.
# operator: clicking it toggles its own marker_14<->15 and every tile at one lattice step (+-8 in x or y).
# glyph: inert; glyph neighbours of an operator are untouched; clicks on empty cells / non-click actions no-op.
# Unconfirmed: whether operators chain (toggle neighbouring operators) - never observed; assumed not. Stateless.
import copy

STEP = 8
FLIP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}
NEIGHBOURS = [(STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)]


def hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def flip_tags(o):
    o["tags"] = [FLIP.get(t, t) for t in o["tags"]]


def tile_at(state, x, y):
    for o in state:
        if o["type"] == "tile" and o["x"] == x and o["y"] == y:
            return o
    return None


def transition_function(state, action):
    state = copy.deepcopy(state)
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return state
    cx, cy = action["x"], action["y"]
    target = next((o for o in state if o.get("visible", True) and hit(o, cx, cy)), None)
    if target is None:
        return state
    if target["type"] == "tile":
        flip_tags(target)
    elif target["type"] == "operator":
        flip_tags(target)
        for dx, dy in NEIGHBOURS:
            t = tile_at(state, target["x"] + dx, target["y"] + dy)
            if t is not None:
                flip_tags(t)
    return state
