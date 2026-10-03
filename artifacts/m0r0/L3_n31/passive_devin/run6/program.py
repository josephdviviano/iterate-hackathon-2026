# Mechanics: two cyan blocks (block mode) move together: A1/A2 y-/+4, A3 apart (left x-4, right x+4), A4 together; each
# block moves alone if its target 4x4 cell is in bounds [2,58], not a maze cell and not a marker cell (marker xy-1).
# Clicking a marker arms: blocks -> color_1 players, marker -> active color_11; A1-A4 then move the active marker by 4
# (blocked by maze/other markers/players); clicking a player disarms; other clicks/A5/blocked moves are no-ops.
# Timer bars (color_0, top-right & bottom-left) w=floor(3(a+1)/7), a=actions since level start (hidden; fallback from w). Maze cells inferred, unseen ones unknown.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
LO, HI = 2, 58
_last = {"out": None, "a": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_w(a):
    return (3 * (a + 1)) // 7


def _is_bar(o):
    return o["type"] == "wall" and "color_0" in o["tags"]


def _parse(state):
    m = {"blocks": [], "players": [], "markers": [], "scenery": [], "bar": 0}
    for o in state:
        if o["type"] == "player":
            (m["blocks"] if o["name"].startswith("block_") else m["players"]).append([o["x"], o["y"]])
        elif o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif _is_bar(o):
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["scenery"].append(dict(o))
    m["blocks"].sort()
    m["players"].sort()
    return m


def _free(cell, blockers):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in blockers


def _hit(px, py, x, y, size):
    return x <= px < x + size and y <= py < y + size


def _step(m, action):
    marker_cells = {(mx - 1, my - 1) for mx, my, _ in m["markers"]}
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for mk in m["markers"]:
            if _hit(cx, cy, mk[0], mk[1], 2) and not mk[2]:
                for other in m["markers"]:
                    other[2] = other is mk
                if m["blocks"]:
                    m["players"], m["blocks"] = m["blocks"], []
                return
        if any(_hit(cx, cy, px, py, 4) for px, py in m["players"]):
            m["blocks"], m["players"] = sorted(m["players"]), []
            for mk in m["markers"]:
                mk[2] = False
        return
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
    if d is None:
        return
    dx, dy = d
    if m["blocks"]:
        left, right = m["blocks"]
        moves = [(left, dx, dy), (right, -dx if dx else 0, dy)]
        for b, bx, by in moves:
            tgt = (b[0] + bx, b[1] + by)
            if _free(tgt, marker_cells):
                b[0], b[1] = tgt
        return
    act = [mk for mk in m["markers"] if mk[2]]
    if not act:
        return
    mk = act[0]
    blockers = {(o[0] - 1, o[1] - 1) for o in m["markers"] if o is not mk}
    blockers |= {(px, py) for px, py in m["players"]}
    tgt = (mk[0] - 1 + dx, mk[1] - 1 + dy)
    if _free(tgt, blockers):
        mk[0], mk[1] = tgt[0] + 1, tgt[1] + 1


def _render(m, a):
    objs = []  # (key, type, color, obj without name)
    for o in m["scenery"]:
        color = int(o["tags"][0].split("_")[1])
        objs.append((o, o["type"], color))
    w = _bar_w(a)
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append(({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                          "visible": True, "w": w, "x": x, "y": y}, "wall", 0))
    for x, y, active in m["markers"]:
        c = 11 if active else 9
        objs.append(({"h": 2, "layer": 1, "tags": ["color_%d" % c, "active" if active else "inactive", "marker"],
                      "type": "marker", "visible": True, "w": 2, "x": x, "y": y}, "marker", c))
    for x, y in m["players"]:
        objs.append(({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                      "visible": True, "w": 4, "x": x, "y": y}, "player", 1))
    out = []
    base = 0
    if m["blocks"]:
        base = 2
        for name, (x, y) in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"h": 4, "layer": 1, "name": name, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": x, "y": y})
    objs.sort(key=lambda t: (t[0]["y"], t[0]["x"]))
    for i, (o, typ, c) in enumerate(objs):
        o = dict(o)
        o["name"] = "%s_%d_%d" % (typ, c, base + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = _parse(state)
    if _last["out"] is not None and _canon(state) == _last["out"]:
        a = _last["a"]
    else:
        a = (7 * m["bar"] - 1) // 3 if m["bar"] > 0 else 0
    _step(m, action)
    a += 1
    out = _render(m, a)
    _last["out"], _last["a"] = _canon(out), a
    return out
