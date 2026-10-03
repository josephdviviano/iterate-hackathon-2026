# Mechanics: block mode = two cyan blocks (block_left/right); ACTION1/2 move both up/down 4, ACTION3/4 move them
# apart/together (mirrored), each blocked independently by hidden maze cells and marker cells (marker pos - 1).
# Clicking a marker arms the blocks (color_1 players) and activates it; ACTION1-4 then move the active marker by 4;
# clicking another marker switches, the active one is a no-op, clicking a player returns to block mode. ACTION5 no-op.
# Every action bumps hidden n; bars wall_0 at top-right/bottom-left have width floor(3(n+1)/7). Unconfirmed: board bounds, ACTION7.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
LO, HI = 2, 58
_mem = {"state": None, "n": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _color(o):
    for t in o["tags"]:
        if t.startswith("color_"):
            return int(t[6:])
    return None


def bar_width(n):
    return 3 * (n + 1) // 7


def _n_from_width(w):
    n = 0
    while bar_width(n) < w:
        n += 1
    return n


def _parse(state):
    g = {"blocks": None, "players": [], "markers": [], "active": None, "bar": 0, "static": []}
    blocks = {}
    for o in state:
        t = o["type"]
        if t == "player" and "block" in o["tags"]:
            blocks[o["name"]] = (o["x"], o["y"])
        elif t == "player":
            g["players"].append((o["x"], o["y"]))
        elif t == "marker":
            g["markers"].append((o["x"], o["y"]))
            if "active" in o["tags"]:
                g["active"] = (o["x"], o["y"])
        elif t == "wall" and _color(o) == 0:
            g["bar"] = max(g["bar"], o["w"])
        else:
            g["static"].append(o)
    if blocks:
        g["blocks"] = [blocks["block_left"], blocks["block_right"]]
    return g


def _free(cell, occupied):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in occupied


def _marker_cells(g, skip=None):
    return {(x - 1, y - 1) for (x, y) in g["markers"] if (x, y) != skip}


def move_blocks(g, action):
    dy = {1: -4, 2: 4}.get(action, 0)
    dxl = {3: -4, 4: 4}.get(action, 0)
    occ = _marker_cells(g)
    out = []
    for i, (x, y) in enumerate(g["blocks"]):
        dx = dxl if i == 0 else -dxl
        nxt = (x + dx, y + dy)
        out.append(nxt if _free(nxt, occ) else (x, y))
    g["blocks"] = out


def move_marker(g, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    ax, ay = g["active"]
    occ = _marker_cells(g, skip=(ax, ay)) | set(g["players"])
    if _free((ax - 1 + dx, ay - 1 + dy), occ):
        nxt = (ax + dx, ay + dy)
        g["markers"] = [nxt if m == (ax, ay) else m for m in g["markers"]]
        g["active"] = nxt


def _hit(px, py, x, y, w, h):
    return x <= px < x + w and y <= py < y + h


def click(g, px, py):
    marker = next((m for m in g["markers"] if _hit(px, py, m[0], m[1], 2, 2)), None)
    if g["blocks"] is not None:
        if marker is not None:
            g["players"] = list(g["blocks"])
            g["blocks"] = None
            g["active"] = marker
        return
    if marker is not None:
        g["active"] = marker
        return
    if any(_hit(px, py, x, y, 4, 4) for (x, y) in g["players"]):
        g["blocks"] = sorted(g["players"])
        g["players"] = []
        g["active"] = None


def _render(g, n):
    objs = []
    w = bar_width(n)
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": w, "x": x, "y": y})
    for o in g["static"]:
        o = dict(o)
        o.pop("name", None)
        objs.append(o)
    for m in g["markers"]:
        act = m == g["active"]
        objs.append({"h": 2, "layer": 1, "type": "marker", "visible": True, "w": 2, "x": m[0], "y": m[1],
                     "tags": ["color_11", "active", "marker"] if act else ["color_9", "inactive", "marker"]})
    for p in g["players"]:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                     "visible": True, "w": 4, "x": p[0], "y": p[1]})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    base = 0
    out = []
    if g["blocks"] is not None:
        base = 2
        for name, (x, y) in zip(("block_left", "block_right"), g["blocks"]):
            out.append({"h": 4, "layer": 1, "name": name, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": x, "y": y})
    for i, o in enumerate(objs):
        o["name"] = "%s_%d_%d" % (o["type"], _color(o), base + i)
        out.append(o)
    return out


def transition_function(state, action):
    g = _parse(state)
    if _mem["state"] is not None and _canon(state) == _mem["state"]:
        n = _mem["n"]
    else:
        n = _n_from_width(g["bar"])
    if isinstance(action, dict):
        click(g, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if g["blocks"] is not None:
            move_blocks(g, action)
        elif g["active"] is not None:
            move_marker(g, action)
    n += 1
    out = _render(g, n)
    _mem["state"], _mem["n"] = _canon(out), n
    return out
