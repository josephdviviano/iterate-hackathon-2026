# Mechanics: lights-out board. Tiles (editable, color_14/15) toggle colour when a click lands in their bbox.
# Operators (named glyph_r_c, tag marker_14/15) toggle their own marker and the orthogonal tiles one lattice
# step (8 cells) away when clicked; glyph neighbours are untouched. Glyphs are inert; empty clicks no-op.
# Non-click actions are no-ops. Stateless: no counter/HUD objects exist in this schema.
# Unconfirmed: operator neighbourhood beyond the 4 orthogonal tiles, and diagonal effects (never observed).
import copy

STEP = 8
TOGGLE = {"color_14": "color_15", "color_15": "color_14",
          "marker_14": "marker_15", "marker_15": "marker_14"}


def hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def toggle(o):
    o["tags"] = [TOGGLE.get(t, t) for t in o["tags"]]


def tile_at(state, x, y):
    for o in state:
        if o["type"] == "tile" and o["x"] == x and o["y"] == y:
            return o
    return None


def click_tile(state, o):
    toggle(o)


def click_operator(state, o):
    toggle(o)
    for dx, dy in ((STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)):
        t = tile_at(state, o["x"] + dx, o["y"] + dy)
        if t is not None:
            toggle(t)


RULES = {"tile": click_tile, "operator": click_operator}


def transition_function(state, action):
    state = copy.deepcopy(state)
    if not isinstance(action, dict) or action.get("action_id") != 6:
        return state
    x, y = action["x"], action["y"]
    for o in state:
        if o.get("visible", True) and hit(o, x, y) and o["type"] in RULES:
            RULES[o["type"]](state, o)
            break
    return state
