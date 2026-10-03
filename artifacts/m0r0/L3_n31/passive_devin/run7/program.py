# Mechanics: mirror-blocks/marker game. Block mode: A1/A2 move both blocks y-4/y+4, A3/A4 move
# block_left by -4/+4 and block_right by +4/-4 in x; each block moves alone into a free 4x4 cell
# (bounds [2,58], invisible maze cells, marker cells). Click marker -> armed mode (players, active
# marker moves with A1-4); click player -> block mode. Hidden: action counter a (bar w=3(a+1)//7).
# Unconfirmed: maze cells beyond those inferred, marker blocking rules, level end/reset triggers.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
DELTA = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"out": None, "a": 0, "left": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def bar_width(a):
    return 3 * (a + 1) // 7


def fallback_a(w):
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def parse(state):
    model = {"bodies": [], "markers": [], "scenery": [], "bar": 0, "armed": False}
    for o in state:
        tags = o.get("tags", [])
        if o["type"] == "player":
            model["bodies"].append({"x": o["x"], "y": o["y"], "name": o["name"]})
            if "armed" in tags:
                model["armed"] = True
        elif o["type"] == "marker":
            model["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in tags})
        elif "color_0" in tags:
            model["bar"] = max(model["bar"], o["w"])
        else:
            model["scenery"].append(o)
    return model


def assign_sides(bodies, remembered):
    if not model_armed_names(bodies):
        left = [b for b in bodies if b["name"] == "block_left"]
        right = [b for b in bodies if b["name"] == "block_right"]
        if left and right:
            return left[0], right[0]
    if remembered is not None:
        for i, b in enumerate(bodies):
            if (b["x"], b["y"]) == remembered and len(bodies) == 2:
                return b, bodies[1 - i]
    s = sorted(bodies, key=lambda b: (b["x"], b["y"]))
    return s[0], s[-1]


def model_armed_names(bodies):
    return any(b["name"].startswith("player") for b in bodies)


def cell_free(cell, blocked):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in blocked


def marker_cell(m):
    return (m["x"] - 1, m["y"] - 1)


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def step(model, action, left, right):
    markers = model["markers"]
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for m in markers:
            if hit(cx, cy, m["x"], m["y"], 2):
                if not m["active"]:
                    for k in markers:
                        k["active"] = k is m
                    model["armed"] = True
                return
        if model["armed"]:
            for b in (left, right):
                if hit(cx, cy, b["x"], b["y"], 4):
                    for k in markers:
                        k["active"] = False
                    model["armed"] = False
                    return
        return
    if action not in DELTA:
        return
    dx, dy = DELTA[action]
    if model["armed"]:
        act = [m for m in markers if m["active"]]
        if not act:
            return
        m = act[0]
        blocked = {marker_cell(k) for k in markers if k is not m}
        blocked |= {(b["x"], b["y"]) for b in (left, right)}
        if cell_free((m["x"] - 1 + dx, m["y"] - 1 + dy), blocked):
            m["x"] += dx
            m["y"] += dy
        return
    mcells = {marker_cell(k) for k in markers}
    for b, sx in ((left, dx), (right, -dx)):
        cell = (b["x"] + sx, b["y"] + dy)
        if cell_free(cell, mcells):
            b["x"], b["y"] = cell


def color_of(o):
    for t in o["tags"]:
        if t.startswith("color_"):
            return t[6:]
    return "0"


def render(model, left, right, a):
    objs = []
    for o in model["scenery"]:
        objs.append(dict(o))
    w = bar_width(a)
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": w, "x": x, "y": y})
    for m in model["markers"]:
        c, s = ("11", "active") if m["active"] else ("9", "inactive")
        objs.append({"h": 2, "layer": 1, "tags": ["color_" + c, s, "marker"], "type": "marker",
                     "visible": True, "w": 2, "x": m["x"], "y": m["y"]})
    blocks = []
    for b, side in ((left, "left"), (right, "right")):
        if model["armed"]:
            objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"],
                         "type": "player", "visible": True, "w": 4, "x": b["x"], "y": b["y"]})
        else:
            blocks.append({"h": 4, "layer": 1, "name": "block_" + side,
                           "pixels": [[10] * 4 for _ in range(4)],
                           "tags": ["cyan", "block", "player"], "type": "player",
                           "visible": True, "w": 4, "x": b["x"], "y": b["y"]})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = 0 if model["armed"] else 2
    for i, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"], color_of(o), start + i)
    return objs + blocks


def transition_function(state, action):
    cont = _mem["out"] is not None and canon(state) == _mem["out"]
    model = parse(state)
    if cont:
        a = _mem["a"]
        remembered = _mem["left"]
    else:
        a = fallback_a(model["bar"])
        remembered = None
    left, right = assign_sides(model["bodies"], remembered)
    step(model, action, left, right)
    a += 1
    out = render(model, left, right, a)
    _mem.update(out=canon(out), a=a, left=(left["x"], left["y"]))
    return out
