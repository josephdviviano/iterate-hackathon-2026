# Mechanics: lights-out board on an 8-cell grid; only ACTION6 clicks act (others no-op).
# Blocking guard first: a click that hits no tile/operator bbox (empty cell or inert glyph) changes nothing.
# Tile rule: clicked editable tile toggles color_14 <-> color_15.
# Operator rule: clicked operator toggles marker_14 <-> marker_15 and toggles tiles at (+-8,0)/(0,+-8); glyphs untouched.
# Unconfirmed: whether operators can toggle other operators or chain; never observed. Stateless.
import copy

STEP = 8
SWAP = {"color_14": "color_15", "color_15": "color_14",
        "marker_14": "marker_15", "marker_15": "marker_14"}


def hit(o, x, y):
    return o.get("visible", True) and o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def toggle(o):
    o["tags"] = [SWAP.get(t, t) for t in o["tags"]]


def click_target(state, x, y):
    for o in state:
        if o["type"] in ("tile", "operator") and hit(o, x, y):
            return o
    return None


def transition_function(state, action):
    state = copy.deepcopy(state)
    if not (isinstance(action, dict) and action.get("action_id") == 6):
        return state
    target = click_target(state, action["x"], action["y"])
    if target is None:
        return state
    toggle(target)
    if target["type"] == "operator":
        nbrs = {(target["x"] + dx, target["y"] + dy)
                for dx, dy in ((STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP))}
        for o in state:
            if o["type"] == "tile" and (o["x"], o["y"]) in nbrs:
                toggle(o)
    return state
