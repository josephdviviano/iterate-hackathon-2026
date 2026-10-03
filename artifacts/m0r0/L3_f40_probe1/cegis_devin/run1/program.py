# Mechanics: two cyan blocks move together (A1/A2 y-+4; A3 apart, A4 together, mirrored); each block stops alone if its
# target 4x4 cell leaves [2,58], hits an invisible maze cell, a marker cell (marker xy-1) or the other block. Clicking a marker
# arms (blocks -> color_1 players, marker -> active color_11); armed A1-4 steer the active marker +-4 under the same blocking;
# click another marker = switch, click a player = disarm, other clicks/A5 no-op. Bars w=floor(3(a+1)/7), a = actions this level.
# Unconfirmed: maze cells only inferred from failed moves ((22,18) from step 124); stateless a = 0 if no bar else top of band.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
_last = {"canon": None, "a": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(a):
    return (3 * (a + 1)) // 7


def fallback_a(w):
    if w <= 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def overlap(ax, ay, bx, by, s=4):
    return abs(ax - bx) < s and abs(ay - by) < s


def parse(state):
    m = {"blocks": {}, "players": [], "markers": [], "scenery": [], "w": 0}
    for o in state:
        tags = o.get("tags", [])
        if o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in tags])
        elif o["name"] in ("block_left", "block_right"):
            m["blocks"][o["name"]] = [o["x"], o["y"]]
        elif o["type"] == "player":
            m["players"].append([o["x"], o["y"]])
        elif tags and tags[0] == "color_0":
            if o["y"] == 63:
                m["w"] = o["w"]
        else:
            m["scenery"].append(o)
    return m


def cell_free(cx, cy, obstacles):
    if not (LO <= cx <= HI and LO <= cy <= HI):
        return False
    if any(overlap(cx, cy, mx, my) for mx, my in MAZE):
        return False
    return not any(overlap(cx, cy, ox, oy) for ox, oy in obstacles)


def step_blocks(m, action):
    d = {1: (0, -4, 0, -4), 2: (0, 4, 0, 4), 3: (-4, 0, 4, 0), 4: (4, 0, -4, 0)}[action]
    bl, br = m["blocks"]["block_left"], m["blocks"]["block_right"]
    mcells = [(x - 1, y - 1) for x, y, _ in m["markers"]]
    nl = [bl[0] + d[0], bl[1] + d[1]]
    nr = [br[0] + d[2], br[1] + d[3]]
    okl = cell_free(nl[0], nl[1], mcells + [tuple(br)])
    okr = cell_free(nr[0], nr[1], mcells + [tuple(bl)])
    if okl and okr and overlap(nl[0], nl[1], nr[0], nr[1]):
        okl = okr = False
    if okl:
        m["blocks"]["block_left"] = nl
    if okr:
        m["blocks"]["block_right"] = nr


def step_marker(m, action):
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    for mk in m["markers"]:
        if mk[2]:
            obst = [(x - 1, y - 1) for x, y, a in m["markers"] if not a] + [tuple(p) for p in m["players"]]
            nx, ny = mk[0] + d[0], mk[1] + d[1]
            if cell_free(nx - 1, ny - 1, obst):
                mk[0], mk[1] = nx, ny
            return


def hit(cx, cy, x, y, s):
    return x <= cx < x + s and y <= cy < y + s


def click(m, cx, cy):
    armed = not m["blocks"]
    for mk in m["markers"]:
        if hit(cx, cy, mk[0], mk[1], 2):
            if mk[2]:
                return
            for o in m["markers"]:
                o[2] = False
            mk[2] = True
            if not armed:
                m["players"] = sorted(m["blocks"].values())
                m["blocks"] = {}
            return
    if armed and any(hit(cx, cy, p[0], p[1], 4) for p in m["players"]):
        ps = sorted(m["players"])
        m["blocks"] = {"block_left": ps[0], "block_right": ps[1]}
        m["players"] = []
        for o in m["markers"]:
            o[2] = False


def render(m, w):
    items = []
    if w > 0:
        items.append({"type": "wall", "tags": ["color_0", "wall"], "x": 64 - w, "y": 0, "w": w, "h": 1, "layer": 0})
        items.append({"type": "wall", "tags": ["color_0", "wall"], "x": 0, "y": 63, "w": w, "h": 1, "layer": 0})
    for o in m["scenery"]:
        o = dict(o)
        o.pop("name", None)
        items.append(o)
    for x, y, act in m["markers"]:
        col = "color_11" if act else "color_9"
        items.append({"type": "marker", "tags": [col, "active" if act else "inactive", "marker"],
                      "x": x, "y": y, "w": 2, "h": 2, "layer": 1})
    for x, y in m["players"]:
        items.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": x, "y": y, "w": 4, "h": 4, "layer": 1})
    out = []
    rank = 2 if m["blocks"] else 0
    for o in sorted(items, key=lambda o: (o["y"], o["x"])):
        o.setdefault("visible", True)
        o["name"] = "%s_%s_%d" % (o["type"], o["tags"][0].split("_", 1)[1], rank)
        rank += 1
        out.append(o)
    for name, (x, y) in m["blocks"].items():
        out.append({"name": name, "type": "player", "tags": ["cyan", "block", "player"], "x": x, "y": y,
                    "w": 4, "h": 4, "layer": 1, "visible": True, "pixels": [r[:] for r in BLOCK_PIX]})
    return out


def transition_function(state, action):
    m = parse(state)
    c = _canon(state)
    a = _last["a"] if c == _last["canon"] and _last["a"] is not None else fallback_a(m["w"])
    if isinstance(action, dict):
        click(m, action.get("x"), action.get("y"))
    elif action in (1, 2, 3, 4):
        if m["blocks"]:
            step_blocks(m, action)
        else:
            step_marker(m, action)
    a += 1
    out = render(m, bar_width(a))
    _last["canon"], _last["a"] = _canon(out), a
    return out
