# Mechanics: two cyan blocks (block mode) move y-/+4 on A1/A2 and mirrored apart/together on A3/A4; each block
# moves alone unless its target 4x4 cell is out of [2,58], a hidden maze cell, a marker cell (marker xy-1) or the
# other block. Clicking an inactive marker arms: blocks -> color_1 players, marker -> active color_11 which A1-A4 move.
# Clicking a player disarms; clicks elsewhere / on the active marker are no-ops. Timer bar w=3(a+1)//7, a=actions.
# Hypothesis: invisible maze {(14,38),(6,18),(38,14),(22,18)}; the (22,18) cell only inferred from step 124 (A4).
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
PIX = [[10] * 4 for _ in range(4)]
_mem = {"last": None, "a": 0, "left": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def parse(state):
    m = {"blocks": {}, "players": [], "markers": [], "active": None, "scenery": [], "bar": 0}
    for o in state:
        t = o["type"]
        if t == "player" and "block" in o["tags"]:
            m["blocks"][o["name"]] = (o["x"], o["y"])
        elif t == "player":
            m["players"].append((o["x"], o["y"]))
        elif t == "marker":
            m["markers"].append((o["x"], o["y"]))
            if "active" in o["tags"]:
                m["active"] = (o["x"], o["y"])
        elif t == "wall" and "color_0" in o["tags"]:
            m["bar"] = o["w"]
        else:
            m["scenery"].append(o)
    return m


def cell_free(c, blocked):
    return LO <= c[0] <= HI and LO <= c[1] <= HI and c not in MAZE and c not in blocked


def marker_cells(markers, skip=None):
    return {(x - 1, y - 1) for (x, y) in markers if (x, y) != skip}


def step_blocks(m, act):
    dy = {1: -4, 2: 4}.get(act, 0)
    dx = {3: -4, 4: 4}.get(act, 0)
    if not (dx or dy):
        return
    L, R = m["blocks"]["block_left"], m["blocks"]["block_right"]
    mc = marker_cells(m["markers"])
    nl = (L[0] + dx, L[1] + dy)
    nr = (R[0] - dx, R[1] + dy)
    if cell_free(nl, mc | {R}):
        m["blocks"]["block_left"] = nl
    if cell_free(nr, mc | {L}):
        m["blocks"]["block_right"] = nr


def step_marker(m, act):
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(act)
    if d is None or m["active"] is None:
        return
    ax, ay = m["active"]
    nx, ny = ax + d[0], ay + d[1]
    blocked = marker_cells(m["markers"], skip=m["active"]) | set(m["players"])
    if cell_free((nx - 1, ny - 1), blocked):
        i = m["markers"].index(m["active"])
        m["markers"][i] = (nx, ny)
        m["active"] = (nx, ny)


def hit(px, py, x, y, s):
    return x <= px < x + s and y <= py < y + s


def click(m, px, py, left):
    for mk in m["markers"]:
        if hit(px, py, mk[0], mk[1], 2):
            if mk == m["active"]:
                return left
            if m["blocks"]:
                L = m["blocks"]["block_left"]
                m["players"] = [L, m["blocks"]["block_right"]]
                m["blocks"] = {}
                left = L
            m["active"] = mk
            return left
    if m["players"]:
        for p in m["players"]:
            if hit(px, py, p[0], p[1], 4):
                ps = sorted(m["players"])
                L = left if left in ps else ps[0]
                R = [p2 for p2 in ps if p2 != L][0] if len(ps) > 1 else L
                m["blocks"] = {"block_left": L, "block_right": R}
                m["players"] = []
                m["active"] = None
                return None
    return left


def render(m, a):
    objs = []
    for o in m["scenery"]:
        o = dict(o)
        objs.append(o)
    w = 3 * (a + 1) // 7
    if w > 0:
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                     "w": w, "x": 64 - w, "y": 0, "_c": "0"})
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                     "w": w, "x": 0, "y": 63, "_c": "0"})
    for (x, y) in m["markers"]:
        act = (x, y) == m["active"]
        c = "11" if act else "9"
        objs.append({"h": 2, "layer": 1, "tags": ["color_" + c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": x, "y": y, "_c": c})
    for (x, y) in m["players"]:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                     "visible": True, "w": 4, "x": x, "y": y, "_c": "1"})
    start = 2 if m["blocks"] else 0
    objs.sort(key=lambda o: (o["y"], o["x"]))
    out = []
    for i, o in enumerate(objs):
        c = o.pop("_c", None)
        if c is None:
            c = [t for t in o["tags"] if t.startswith("color_")][0][6:]
        o["name"] = "%s_%s_%d" % (o["type"], c, start + i)
        out.append(o)
    for name, (x, y) in m["blocks"].items():
        out.append({"h": 4, "layer": 1, "name": name, "pixels": [r[:] for r in PIX],
                    "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                    "w": 4, "x": x, "y": y})
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["last"] is not None and canon(state) == _mem["last"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] else 0
        left = min(m["players"]) if m["players"] else None
    if isinstance(action, dict):
        left = click(m, action["x"], action["y"], left)
    elif m["blocks"]:
        step_blocks(m, action)
    else:
        step_marker(m, action)
    a += 1
    out = render(m, a)
    _mem.update(last=canon(out), a=a, left=left)
    return out
