# Mechanics: mirror-blocks game. Block mode: A1/A2 move both cyan blocks y-4/y+4, A3 apart, A4 together
# (left gets dx, right -dx); each block blocked by bounds [2,58], the other block, marker cells and an
# invisible maze {(14,38),(6,18),(38,14),(22,18)}. Click marker -> armed (blocks become color_1 players,
# marker active color_11); armed A1-4 move active marker by 4; click player -> back to blocks. Global HUD:
# bar w=3(a+1)//7, a = actions since level start (hidden, continuity-gated; fallback ceil(7w/3)). Names re-rank by (y,x).
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
_mem = {"canon": None, "a": 0, "left": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _parse(state):
    m = {"armed": False, "blocks": {}, "players": [], "markers": [], "bar": 0, "bg": []}
    for o in state:
        t, tags = o["type"], o.get("tags", [])
        if t == "player" and "armed" in tags:
            m["armed"] = True
            m["players"].append((o["x"], o["y"]))
        elif t == "player":
            m["blocks"]["left" if o["name"] == "block_left" else "right"] = [o["x"], o["y"]]
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in tags])
        elif t == "wall" and "color_0" in tags:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["bg"].append(o)
    return m


def _marker_cells(m, skip=None):
    return {(mk[0] - 1, mk[1] - 1) for i, mk in enumerate(m["markers"]) if i != skip}


def _free(cell, blocked):
    return 2 <= cell[0] <= 58 and 2 <= cell[1] <= 58 and cell not in blocked and cell not in MAZE


def _move_blocks(m, action):
    dy = {1: -4, 2: 4}.get(action, 0)
    dx = {3: -4, 4: 4}.get(action, 0)
    if not (dx or dy):
        return
    mc = _marker_cells(m)
    for side, sdx in (("left", dx), ("right", -dx)):
        b = m["blocks"][side]
        other = m["blocks"]["right" if side == "left" else "left"]
        nc = (b[0] + sdx, b[1] + dy)
        if _free(nc, mc | {tuple(other)}):
            b[0], b[1] = nc


def _move_marker(m, action):
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
    if d is None:
        return
    for i, mk in enumerate(m["markers"]):
        if mk[2]:
            nc = (mk[0] - 1 + d[0], mk[1] - 1 + d[1])
            blocked = _marker_cells(m, skip=i) | set(m["players"])
            if _free(nc, blocked):
                mk[0], mk[1] = nc[0] + 1, nc[1] + 1


def _hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def _click(m, x, y, left):
    hit_m = next((i for i, mk in enumerate(m["markers"]) if _hit(x, y, mk[0], mk[1], 2)), None)
    if m["armed"]:
        if any(_hit(x, y, px, py, 4) for px, py in m["players"]):
            ps = sorted(m["players"])
            lp = left if left in ps else ps[0]
            rp = [p for p in ps if p != lp][0] if len(ps) > 1 else lp
            m["armed"] = False
            m["blocks"] = {"left": list(lp), "right": list(rp)}
            m["players"] = []
            for mk in m["markers"]:
                mk[2] = False
        elif hit_m is not None:
            for i, mk in enumerate(m["markers"]):
                mk[2] = i == hit_m
    elif hit_m is not None:
        m["armed"] = True
        left = tuple(m["blocks"]["left"])
        m["players"] = [left, tuple(m["blocks"]["right"])]
        m["blocks"] = {}
        for i, mk in enumerate(m["markers"]):
            mk[2] = i == hit_m
    return left


def _render(m, a):
    w = 3 * (a + 1) // 7
    objs = []
    if w > 0:
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                     "w": w, "x": 64 - w, "y": 0, "_c": "0"})
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                     "w": w, "x": 0, "y": 63, "_c": "0"})
    for o in m["bg"]:
        o = dict(o)
        o["_c"] = o["name"].split("_")[1]
        objs.append(o)
    for x, y, act in m["markers"]:
        c = "11" if act else "9"
        objs.append({"h": 2, "layer": 1, "tags": ["color_" + c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": x, "y": y, "_c": c})
    for x, y in m["players"]:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                     "visible": True, "w": 4, "x": x, "y": y, "_c": "1"})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    base = 0 if m["armed"] else 2
    out = []
    for i, o in enumerate(objs):
        c = o.pop("_c")
        o["name"] = "%s_%s_%d" % (o["type"], c, base + i)
        out.append(o)
    for side in ("left", "right"):
        if side in m["blocks"]:
            x, y = m["blocks"][side]
            out.append({"h": 4, "layer": 1, "name": "block_" + side, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": x, "y": y})
    return out


def transition_function(state, action):
    m = _parse(state)
    if _mem["canon"] is not None and _canon(state) == _mem["canon"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] else 0
        left = min(m["players"]) if m["players"] else None
    if isinstance(action, dict):
        left = _click(m, action["x"], action["y"], left)
    elif m["armed"]:
        _move_marker(m, action)
    else:
        _move_blocks(m, action)
    a += 1
    out = _render(m, a)
    _mem.update(canon=_canon(out), a=a, left=left)
    return out
