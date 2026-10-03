# Mechanics: mirror-blocks/marker game. Block mode: ACTION1/2 move both cyan blocks y-4/y+4, ACTION3 apart, ACTION4 together;
# each block moves independently, blocked by bounds, maze cells and marker cells. Clicking an inactive marker arms the game:
# blocks -> color_1 players, clicked marker active; ACTION1-4 then move the active marker by 4; clicking a player disarms.
# Timer bars (color_0, top-right and bottom-left) have width floor(3(k+1)/7), k = actions since level start (hidden, continuity
# gated; fallback = smallest k for the bar width). Names = type_color_rank by (y,x); blocks reserve 0,1. Maze cells are inferred.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
STEP = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"canon": None, "k": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_width(k):
    return (3 * (k + 1)) // 7


def _k_from_width(w):
    return 0 if w <= 0 else (7 * w - 1) // 3


def _parse(state):
    m = {"blocks": [], "players": [], "markers": [], "static": [], "bar": 0}
    for o in state:
        tags = o.get("tags", [])
        if "block" in tags:
            m["blocks"].append([o["x"], o["y"]])
        elif "armed" in tags:
            m["players"].append([o["x"], o["y"]])
        elif o["type"] == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in tags})
        elif "color_0" in tags:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["static"].append(dict(o))
    return m


def _cell_free(cell, occupied):
    x, y = cell
    return 2 <= y <= 58 and cell not in MAZE and cell not in occupied


def _block_ok(cell, left, occupied):
    x = cell[0]
    lo, hi = (2, 26) if left else (34, 58)
    return lo <= x <= hi and _cell_free(cell, occupied)


def _marker_cells(m, skip=None):
    return {(mk["x"] - 1, mk["y"] - 1) for mk in m["markers"] if mk is not skip}


def _hit(o, x, y, w, h):
    return o[0] <= x < o[0] + w and o[1] <= y < o[1] + h


def _step(m, action):
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for mk in m["markers"]:
            if mk["x"] <= cx < mk["x"] + 2 and mk["y"] <= cy < mk["y"] + 2:
                if not mk["active"]:
                    for other in m["markers"]:
                        other["active"] = other is mk
                    if m["blocks"]:
                        m["players"], m["blocks"] = m["blocks"], []
                return
        if m["players"] and any(_hit(p, cx, cy, 4, 4) for p in m["players"]):
            m["blocks"], m["players"] = m["players"], []
            for mk in m["markers"]:
                mk["active"] = False
        return
    if action not in STEP:
        return
    dx, dy = STEP[action]
    if m["blocks"]:
        occ = _marker_cells(m)
        for b in m["blocks"]:
            left = b[0] < 32
            ddx = dx if left else -dx
            if action in (1, 2):
                ddx = 0
            cell = (b[0] + ddx, b[1] + dy)
            if _block_ok(cell, left, occ):
                b[0], b[1] = cell
        return
    act = [mk for mk in m["markers"] if mk["active"]]
    if not act:
        return
    mk = act[0]
    occ = _marker_cells(m, skip=mk) | {tuple(p) for p in m["players"]}
    cell = (mk["x"] - 1 + dx, mk["y"] - 1 + dy)
    if 2 <= cell[0] <= 58 and _cell_free(cell, occ):
        mk["x"], mk["y"] = cell[0] + 1, cell[1] + 1


def _render(m, k):
    objs = []
    w = _bar_width(k)
    bar = {"layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True, "h": 1, "w": w}
    if w > 0:
        objs.append(dict(bar, x=64 - w, y=0))
        objs.append(dict(bar, x=0, y=63))
    for o in m["static"]:
        objs.append(o)
    for mk in m["markers"]:
        c = "color_11" if mk["active"] else "color_9"
        objs.append({"h": 2, "w": 2, "layer": 1, "type": "marker", "visible": True, "x": mk["x"], "y": mk["y"],
                     "tags": [c, "active" if mk["active"] else "inactive", "marker"]})
    for p in m["players"]:
        objs.append({"h": 4, "w": 4, "layer": 1, "type": "player", "visible": True, "x": p[0], "y": p[1],
                     "tags": ["color_1", "armed", "player"]})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    base = 2 if m["blocks"] else 0
    for i, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"], o["tags"][0].split("_")[1], base + i)
    for b in m["blocks"]:
        objs.append({"h": 4, "w": 4, "layer": 1, "type": "player", "visible": True, "x": b[0], "y": b[1],
                     "name": "block_left" if b[0] < 32 else "block_right",
                     "pixels": [[10] * 4 for _ in range(4)], "tags": ["cyan", "block", "player"]})
    return objs


def transition_function(state, action):
    m = _parse(state)
    if _last["canon"] is not None and _canon(state) == _last["canon"]:
        k = _last["k"]
    else:
        k = _k_from_width(m["bar"])
    _step(m, action)
    k += 1
    out = _render(m, k)
    _last["canon"], _last["k"] = _canon(out), k
    return out
