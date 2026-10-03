# Mechanics: BLOCK mode: block_left/right step 4px (1/2 both up/down; 3/4 move left block left/right, right block mirrored);
# a block can't enter a cell overlapping a marker or leave the board. Clicking a marker -> PLAYER mode (blocks become color_1
# 'armed' players, clicked marker active/11); 1-4 move the active marker, click marker = select, click player = back to BLOCK.
# Step bar (color-0 walls top-right/bottom-left) width ceil(3n/7), first action after level start uncounted; names ranked by (y,x).
# Unconfirmed: invisible maze walls inside wall_15/hazard_8 (no pixels in schema) block some moves; not modelled.
import math

DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"out": None, "n": None, "bar": None}


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def overlaps(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def parse(state):
    m = {"blocks": [], "players": [], "markers": [], "static": [], "bar": 0}
    for o in state:
        if o["type"] == "player":
            (m["blocks"] if o["name"].startswith("block_") else m["players"]).append([o["x"], o["y"]])
        elif o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif o["type"] == "wall" and color_of(o) == 0:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["static"].append(dict(o))
    m["mode"] = "player" if m["players"] else "block"
    return m


def bar_width(n):
    return max(0, math.ceil(3 * n / 7))


def counter(state, m):
    if _last["out"] is not None and _last["bar"] == m["bar"] and _last["n"] is not None:
        return _last["n"]
    if m["bar"] == 0:
        return -1
    n = 0
    while bar_width(n) < m["bar"]:
        n += 1
    return n


def free_for_block(m, x, y, other):
    if x < 0 or x + 4 > 64 or y < 1 or y + 4 > 63:
        return False
    if any(overlaps(x, y, 4, 4, mx, my, 2, 2) for mx, my, _ in m["markers"]):
        return False
    return not overlaps(x, y, 4, 4, other[0], other[1], 4, 4)


def free_for_marker(m, i, x, y):
    if x < 0 or x + 2 > 64 or y < 1 or y + 2 > 63:
        return False
    return not any(j != i and overlaps(x, y, 2, 2, mx, my, 2, 2) for j, (mx, my, _) in enumerate(m["markers"]))


def step_block(m, action):
    blocks = sorted(m["blocks"])
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for k, (mx, my, _) in enumerate(m["markers"]):
            if mx <= cx < mx + 2 and my <= cy < my + 2:
                m["mode"], m["players"], m["blocks"] = "player", blocks, []
                for j, mk in enumerate(m["markers"]):
                    mk[2] = j == k
                return
        return
    if action not in DIRS or len(blocks) != 2:
        return
    dx, dy = DIRS[action]
    moves = [(dx, dy), (-dx, dy)]  # right block mirrors horizontally
    old = [list(b) for b in blocks]
    for k in range(2):
        nx, ny = old[k][0] + moves[k][0], old[k][1] + moves[k][1]
        if free_for_block(m, nx, ny, blocks[1 - k]):
            blocks[k] = [nx, ny]
    m["blocks"] = blocks


def step_player(m, action):
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for px, py in m["players"]:
            if px <= cx < px + 4 and py <= cy < py + 4:
                m["mode"], m["blocks"], m["players"] = "block", sorted(m["players"]), []
                for mk in m["markers"]:
                    mk[2] = False
                return
        for k, (mx, my, _) in enumerate(m["markers"]):
            if mx <= cx < mx + 2 and my <= cy < my + 2:
                for j, mk in enumerate(m["markers"]):
                    mk[2] = j == k
                return
        return
    if action not in DIRS:
        return
    dx, dy = DIRS[action]
    for k, mk in enumerate(m["markers"]):
        if mk[2] and free_for_marker(m, k, mk[0] + dx, mk[1] + dy):
            mk[0], mk[1] = mk[0] + dx, mk[1] + dy


def render(m):
    objs = []
    for o in m["static"]:
        objs.append(o)
    w = m["bar"]
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                         "w": w, "x": x, "y": y, "_c": 0})
    for mx, my, act in m["markers"]:
        c = 11 if act else 9
        objs.append({"h": 2, "layer": 1, "tags": ["color_%d" % c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": mx, "y": my, "_c": c})
    for px, py in m["players"]:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                     "visible": True, "w": 4, "x": px, "y": py, "_c": 1})
    out = []
    for bx, by in sorted(m["blocks"]):
        out.append({"h": 4, "layer": 1, "name": "block_left" if bx + 2 <= 32 else "block_right",
                    "pixels": [[10] * 4 for _ in range(4)], "tags": ["cyan", "block", "player"],
                    "type": "player", "visible": True, "w": 4, "x": bx, "y": by})
    base = len(out)
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(objs):
        c = o.pop("_c", None)
        if c is None:
            c = color_of(o)
        o["name"] = "%s_%d_%d" % (o["type"], c, base + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    n = counter(state, m) + 1
    if m["mode"] == "block":
        step_block(m, action)
    else:
        step_player(m, action)
    m["bar"] = bar_width(n)
    out = render(m)
    _last.update(out=out, n=n, bar=m["bar"])
    return [dict(o) for o in out]
