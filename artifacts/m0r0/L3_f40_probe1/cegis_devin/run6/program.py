# Mechanics: two cyan blocks move as a mirrored pair (A1/A2 both y-/+4; A3 left x-4 & right x+4; A4 the reverse),
# each moving alone only if its target 4x4 cell is in bounds [2,58] and not blocked (hidden maze cells, marker cells
# = marker xy-1, the other block). Clicking a marker arms: blocks -> color_1 players, that marker active (11); A1-4 then
# move the active marker by 4 (same blocking); clicking another marker switches, clicking a player disarms. Timer bars
# w=3(a+1)//7, a = actions since level start (continuity-gated). Unconfirmed: maze cell (22,18) only seen at step 124.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
_mem = {"last": None, "a": 0, "left": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def parse(state):
    m = {"blocks": {}, "players": [], "markers": [], "active": None, "scenery": [], "bar": 0}
    for o in state:
        t = o["type"]
        if t == "player" and "block" in o.get("tags", []):
            m["blocks"]["left" if o["name"] == "block_left" else "right"] = (o["x"], o["y"])
        elif t == "player":
            m["players"].append((o["x"], o["y"]))
        elif t == "marker":
            m["markers"].append((o["x"], o["y"]))
            if "active" in o.get("tags", []):
                m["active"] = (o["x"], o["y"])
        elif t == "wall" and color_of(o) == 0:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["scenery"].append(o)
    return m


def in_bounds(c):
    return LO <= c[0] <= HI and LO <= c[1] <= HI


def free(cell, others):
    return in_bounds(cell) and cell not in MAZE and cell not in others


def hit(o_xy, size, x, y):
    return o_xy[0] <= x < o_xy[0] + size and o_xy[1] <= y < o_xy[1] + size


DELTA = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}


def step_blocks(m, act):
    if act not in DELTA:
        return
    dx, dy = DELTA[act]
    marker_cells = {(x - 1, y - 1) for x, y in m["markers"]}
    old = dict(m["blocks"])
    for side, (x, y) in old.items():
        sdx = dx if side == "left" else -dx
        tgt = (x + sdx, y + dy)
        other = {p for s, p in old.items() if s != side}
        if free(tgt, marker_cells | other):
            m["blocks"][side] = tgt


def step_marker(m, act):
    if act not in DELTA or m["active"] is None:
        return
    dx, dy = DELTA[act]
    ax, ay = m["active"]
    tgt = (ax + dx, ay + dy)
    others = {(x - 1, y - 1) for x, y in m["markers"] if (x, y) != m["active"]} | set(m["players"])
    if free((tgt[0] - 1, tgt[1] - 1), others):
        m["markers"] = [tgt if p == m["active"] else p for p in m["markers"]]
        m["active"] = tgt


def click(m, x, y):
    for p in m["markers"]:
        if hit(p, 2, x, y):
            if m["blocks"]:
                m["left"] = m["blocks"]["left"]
                m["players"] = [m["blocks"]["left"], m["blocks"]["right"]]
                m["blocks"] = {}
            m["active"] = p
            return
    if not m["blocks"]:
        for p in m["players"]:
            if hit(p, 4, x, y):
                left = m["left"] if m["left"] in m["players"] else min(m["players"])
                right = [q for q in m["players"] if q != left]
                m["blocks"] = {"left": left, "right": right[0] if right else left}
                m["players"] = []
                m["active"] = None
                return


def render(m, a):
    objs = []
    for o in m["scenery"]:
        objs.append(dict(o))
    for p in m["markers"]:
        act = p == m["active"]
        c = 11 if act else 9
        objs.append({"type": "marker", "tags": ["color_%d" % c, "active" if act else "inactive", "marker"],
                     "x": p[0], "y": p[1], "w": 2, "h": 2, "layer": 1, "visible": True, "_c": c})
    for p in m["players"]:
        objs.append({"type": "player", "tags": ["color_1", "armed", "player"],
                     "x": p[0], "y": p[1], "w": 4, "h": 4, "layer": 1, "visible": True, "_c": 1})
    w = 3 * (a + 1) // 7
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append({"type": "wall", "tags": ["color_0", "wall"], "x": bx, "y": by, "w": w, "h": 1,
                         "layer": 0, "visible": True, "_c": 0})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = 2 if m["blocks"] else 0
    for i, o in enumerate(objs):
        c = o.pop("_c", None)
        if c is None:
            c = color_of(o)
        o["name"] = "%s_%d_%d" % (o["type"], c, start + i)
    for side, p in m["blocks"].items():
        objs.append({"name": "block_" + side, "type": "player", "tags": ["cyan", "block", "player"],
                     "x": p[0], "y": p[1], "w": 4, "h": 4, "layer": 1, "visible": True,
                     "pixels": [row[:] for row in BLOCK_PIX]})
    return objs


def transition_function(state, action):
    m = parse(state)
    if _mem["last"] is not None and canon(state) == _mem["last"]:
        a, m["left"] = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] else 0
        m["left"] = min(m["players"]) if m["players"] else None
    act = action["action_id"] if isinstance(action, dict) else action
    if act == 6:
        click(m, action["x"], action["y"])
    elif m["blocks"]:
        step_blocks(m, act)
    else:
        step_marker(m, act)
    a += 1
    out = render(m, a)
    _mem.update(last=canon(out), a=a, left=m["left"])
    return out
