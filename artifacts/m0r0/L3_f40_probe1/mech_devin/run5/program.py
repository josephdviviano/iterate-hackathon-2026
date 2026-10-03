# Mechanics: two mirrored cyan blocks; A1/A2 move both y-4/+4, A3 left x-4 & right x+4 (A4 reverse); each block moves alone if its target cell is free.
# Blocked cells: bounds [2,58], invisible maze {(14,38),(6,18),(38,14),(22,18)}, marker cells (marker xy-1), the other block. Click marker = arm (blocks -> color_1 players, marker active).
# Armed: A1-4 steer the active marker by 4 (blocked by maze, markers, players); click inactive marker = switch, click player = disarm, other clicks/A5 = no-op.
# Timer bars (color_0 at top-right and bottom-left): w = 3(n+1)//7, n = actions since level start incl. current; names = type_color_rank by (y,x), from 2 in block mode.
# Unconfirmed: maze cells beyond the 4 seen (esp. (22,18), inferred from step 124 alone); counter fallback when continuity breaks = 0 if no bar else largest n for that width.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
_mem = {"canon": None, "n": 0, "left": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return t[6:]
    return None


def parse(state):
    m = {"blocks": None, "players": [], "markers": [], "bar": 0, "scenery": []}
    blocks = {}
    for o in state:
        if o["name"] in ("block_left", "block_right"):
            blocks[o["name"]] = (o["x"], o["y"])
        elif o["type"] == "player":
            m["players"].append((o["x"], o["y"]))
        elif o["type"] == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o["tags"]})
        elif o["type"] == "wall" and color_of(o) == "0":
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["scenery"].append(o)
    if blocks:
        m["blocks"] = [blocks["block_left"], blocks["block_right"]]
    return m


def free_for_block(cell, other, markers):
    x, y = cell
    if not (LO <= x <= HI and LO <= y <= HI):
        return False
    if cell in MAZE or cell == other:
        return False
    return all((mk["x"] - 1, mk["y"] - 1) != cell for mk in markers)


def move_blocks(m, action):
    deltas = {1: [(0, -4), (0, -4)], 2: [(0, 4), (0, 4)],
              3: [(-4, 0), (4, 0)], 4: [(4, 0), (-4, 0)]}[action]
    cur = list(m["blocks"])
    new = list(cur)
    for i in (0, 1):
        tgt = (cur[i][0] + deltas[i][0], cur[i][1] + deltas[i][1])
        if free_for_block(tgt, cur[1 - i], m["markers"]):
            new[i] = tgt
    m["blocks"] = new


def move_marker(m, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    act = [mk for mk in m["markers"] if mk["active"]]
    if not act:
        return
    mk = act[0]
    cell = (mk["x"] - 1 + dx, mk["y"] - 1 + dy)
    if not (LO <= cell[0] <= HI and LO <= cell[1] <= HI) or cell in MAZE:
        return
    if any((o["x"] - 1, o["y"] - 1) == cell for o in m["markers"] if o is not mk):
        return
    if any(p == cell for p in m["players"]):
        return
    mk["x"] += dx
    mk["y"] += dy


def hit(px, py, x, y, size):
    return x <= px < x + size and y <= py < y + size


def click(m, px, py, left_pos):
    target = None
    for mk in m["markers"]:
        if hit(px, py, mk["x"], mk["y"], 2):
            target = mk
    if m["blocks"] is not None:
        if target is not None:
            left_pos = m["blocks"][0]
            m["players"] = list(m["blocks"])
            m["blocks"] = None
            for mk in m["markers"]:
                mk["active"] = mk is target
        return left_pos
    if target is not None:
        if not target["active"]:
            for mk in m["markers"]:
                mk["active"] = mk is target
        return left_pos
    for p in m["players"]:
        if hit(px, py, p[0], p[1], 4):
            ps = sorted(m["players"])
            left = left_pos if left_pos in ps else ps[0]
            right = [q for q in ps if q != left][0] if len(ps) > 1 else left
            m["blocks"] = [left, right]
            m["players"] = []
            for mk in m["markers"]:
                mk["active"] = False
            return None
    return left_pos


def render(m, bar_w):
    objs = []
    for o in m["scenery"]:
        d = dict(o)
        d["_prefix"] = o["name"].rsplit("_", 1)[0]
        objs.append(d)
    for mk in m["markers"]:
        c = "11" if mk["active"] else "9"
        objs.append({"_prefix": "marker_" + c, "type": "marker",
                     "tags": ["color_" + c, "active" if mk["active"] else "inactive", "marker"],
                     "x": mk["x"], "y": mk["y"], "w": 2, "h": 2, "layer": 1, "visible": True})
    for (x, y) in m["players"]:
        objs.append({"_prefix": "player_1", "type": "player", "tags": ["color_1", "armed", "player"],
                     "x": x, "y": y, "w": 4, "h": 4, "layer": 1, "visible": True})
    if bar_w > 0:
        for (x, y) in ((64 - bar_w, 0), (0, 63)):
            objs.append({"_prefix": "wall_0", "type": "wall", "tags": ["color_0", "wall"],
                         "x": x, "y": y, "w": bar_w, "h": 1, "layer": 0, "visible": True})
    start = 2 if m["blocks"] is not None else 0
    objs.sort(key=lambda o: (o["y"], o["x"]))
    out = []
    for i, o in enumerate(objs):
        o["name"] = "%s_%d" % (o.pop("_prefix"), start + i)
        out.append(o)
    if m["blocks"] is not None:
        for name, (x, y) in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"name": name, "type": "player", "tags": ["cyan", "block", "player"],
                        "x": x, "y": y, "w": 4, "h": 4, "pixels": [r[:] for r in BLOCK_PIX],
                        "layer": 1, "visible": True})
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["canon"] is not None and canon(state) == _mem["canon"]:
        n, left_pos = _mem["n"], _mem["left"]
    else:
        n = 0 if m["bar"] == 0 else (7 * m["bar"] + 3) // 3
        left_pos = None
    if isinstance(action, dict):
        left_pos = click(m, action.get("x"), action.get("y"), left_pos)
    elif action in (1, 2, 3, 4):
        if m["blocks"] is not None:
            move_blocks(m, action)
        else:
            move_marker(m, action)
    n += 1
    out = render(m, 3 * (n + 1) // 7)
    _mem.update(canon=canon(out), n=n, left=left_pos)
    return out
