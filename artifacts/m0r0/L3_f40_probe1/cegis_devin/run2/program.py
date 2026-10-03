# Mechanics: two cyan blocks (block mode) move together on a 4-cell grid: A1/A2 = both y-/+4, A3 = apart (left x-4, right x+4), A4 = together; each
# block moves alone if its target cell is in bounds [2,58] and not a marker cell (marker xy-1), the other block, or an invisible maze cell.
# A6 on an inactive marker arms it: blocks become color_1 players, that marker turns active (color_11) and A1-A4 move it instead; A6 on a player
# disarms; other clicks/A5/A7 are no-ops. Bar width = floor(3(a+1)/7), a = actions since level start (continuity-gated; fallback = top of band).
# Names = type_color_rank by (y,x) (from 2 in block mode). Unconfirmed: maze cell (22,18) is inferred only from step 124 (no visible cause).
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = {"last": None, "a": 0, "order": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def bar_width(a):
    return (3 * (a + 1)) // 7


def counter_from_width(w):
    if w <= 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def parse(state):
    m = {"mode": "armed", "blocks": [], "markers": [], "scenery": [], "bar": 0}
    players = []
    for o in state:
        t, tags = o["type"], o.get("tags", [])
        if t == "player":
            if "block" in tags:
                m["mode"] = "block"
            players.append(o)
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in tags])
        elif "color_0" in tags:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["scenery"].append({k: v for k, v in o.items() if k != "name"})
    if m["mode"] == "block":
        byname = {o["name"]: (o["x"], o["y"]) for o in players}
        m["blocks"] = [byname["block_left"], byname["block_right"]]
    else:
        ps = sorted((o["x"], o["y"]) for o in players)
        m["blocks"] = ps
    return m


def free_cell(cell, others):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in others


def marker_cells(markers, skip=None):
    return {(mx - 1, my - 1) for i, (mx, my, _) in enumerate(markers) if i != skip}


def move_blocks(m, dx, dy):
    signs = (1, 1) if dx == 0 else (1, -1)
    mcells = marker_cells(m["markers"])
    old = list(m["blocks"])
    new = []
    for i, (x, y) in enumerate(old):
        tgt = (x + dx * signs[i], y + dy)
        other = old[1 - i]
        new.append(tgt if free_cell(tgt, mcells | {other}) else (x, y))
    if new[0] == new[1]:
        new = old
    m["blocks"] = new


def move_marker(m, dx, dy):
    for i, mk in enumerate(m["markers"]):
        if mk[2]:
            tgt = (mk[0] - 1 + dx, mk[1] - 1 + dy)
            others = marker_cells(m["markers"], skip=i) | set(m["blocks"])
            if free_cell(tgt, others):
                mk[0], mk[1] = tgt[0] + 1, tgt[1] + 1


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def click(m, cx, cy):
    for i, mk in enumerate(m["markers"]):
        if hit(cx, cy, mk[0], mk[1], 2):
            if not mk[2]:
                for j, other in enumerate(m["markers"]):
                    other[2] = (j == i)
                m["mode"] = "armed"
            return
    if m["mode"] == "armed":
        for bx, by in m["blocks"]:
            if hit(cx, cy, bx, by, 4):
                for mk in m["markers"]:
                    mk[2] = False
                m["mode"] = "block"
                return


def step(m, action):
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
        return
    if action in DIRS:
        dx, dy = DIRS[action]
        if m["mode"] == "block":
            move_blocks(m, dx, dy)
        else:
            move_marker(m, dx, dy)


def render(m, w):
    objs = [dict(o) for o in m["scenery"]]
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"type": "wall", "tags": ["color_0", "wall"], "x": x, "y": y,
                         "w": w, "h": 1, "layer": 0, "visible": True})
    for mx, my, act in m["markers"]:
        tags = ["color_11", "active", "marker"] if act else ["color_9", "inactive", "marker"]
        objs.append({"type": "marker", "tags": tags, "x": mx, "y": my, "w": 2, "h": 2,
                     "layer": 1, "visible": True})
    fixed = []
    if m["mode"] == "block":
        for name, (x, y) in zip(("block_left", "block_right"), m["blocks"]):
            fixed.append({"name": name, "type": "player", "tags": ["cyan", "block", "player"],
                          "x": x, "y": y, "w": 4, "h": 4, "layer": 1, "visible": True,
                          "pixels": [row[:] for row in BLOCK_PIX]})
        start = 2
    else:
        for x, y in m["blocks"]:
            objs.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": x, "y": y,
                         "w": 4, "h": 4, "layer": 1, "visible": True})
        start = 0
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for r, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"], o["tags"][0].split("_")[1], start + r)
    return fixed + objs


def transition_function(state, action):
    m = parse(state)
    if _memo["last"] is not None and canon(state) == _memo["last"]:
        a = _memo["a"]
        if m["mode"] == "armed" and _memo["order"] is not None:
            m["blocks"] = list(_memo["order"])
    else:
        a = counter_from_width(m["bar"])
    step(m, action)
    a += 1
    out = render(m, bar_width(a))
    _memo.update(last=canon(out), a=a, order=list(m["blocks"]))
    return out
