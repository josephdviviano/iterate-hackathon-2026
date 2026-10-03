# Mechanics: block mode - A1/A2 move both blocks y-/+4, A3 apart / A4 together (mirrored), each block
# independently blocked by half bounds, maze cells and marker cells. Click marker -> arm (blocks become
# color_1 players, marker active); armed A1-A4 steer the active marker by 4 (blocked by maze/markers/players);
# click inactive marker = switch, click player = disarm, other clicks/A5 no-op. Timer bars w=floor(3(a+1)/7).
# Hypothesis: 'marker gone under A3' = (y,x) re-ranking when bars appear; maze beyond 3 observed cells unknown.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
CELL = 4
_mem = {"canon": None, "a": 0}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _bar_w(a):
    return (3 * (a + 1)) // 7


def _parse(state):
    m = {"mode": "block", "blocks": [], "markers": [], "scenery": [], "bar": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            if "armed" in o["tags"]:
                m["mode"] = "armed"
            m["blocks"].append([o["x"], o["y"]])
        elif t == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o["tags"]})
        elif "color_0" in o["tags"]:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["scenery"].append(o)
    m["blocks"].sort(key=lambda b: b[0])  # [left, right]
    return m


def _marker_cells(m, skip=None):
    return {(k["x"] - 1, k["y"] - 1) for k in m["markers"] if k is not skip}


def _block_ok(i, cell, m):
    x, y = cell
    lo, hi = (2, 26) if i == 0 else (34, 58)
    if not (lo <= x <= hi and 2 <= y <= 58):
        return False
    return cell not in MAZE and cell not in _marker_cells(m)


def _marker_ok(cell, m, mk):
    x, y = cell
    if not (2 <= x <= 58 and 2 <= y <= 58):
        return False
    if cell in MAZE or cell in _marker_cells(m, skip=mk):
        return False
    return cell not in {tuple(b) for b in m["blocks"]}


def _hit(o_x, o_y, size, cx, cy):
    return o_x <= cx < o_x + size and o_y <= cy < o_y + size


DELTA = {1: (0, -CELL), 2: (0, CELL), 3: (-CELL, 0), 4: (CELL, 0)}


def _step(m, action):
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for mk in m["markers"]:
            if _hit(mk["x"], mk["y"], 2, cx, cy):
                if not mk["active"]:
                    for k in m["markers"]:
                        k["active"] = k is mk
                    m["mode"] = "armed"
                return
        if m["mode"] == "armed":
            for b in m["blocks"]:
                if _hit(b[0], b[1], 4, cx, cy):
                    m["mode"] = "block"
                    for k in m["markers"]:
                        k["active"] = False
                    return
        return
    if action not in DELTA:
        return
    dx, dy = DELTA[action]
    if m["mode"] == "block":
        moves = [(dx, dy), (-dx, dy)]  # left block, mirrored right block
        for i, b in enumerate(m["blocks"]):
            nx, ny = b[0] + moves[i][0], b[1] + moves[i][1]
            if _block_ok(i, (nx, ny), m):
                b[0], b[1] = nx, ny
    else:
        for mk in m["markers"]:
            if mk["active"]:
                cell = (mk["x"] - 1 + dx, mk["y"] - 1 + dy)
                if _marker_ok(cell, m, mk):
                    mk["x"] += dx
                    mk["y"] += dy


def _render(m, a):
    objs = []  # (sortkey, base dict, prefix or None)
    for o in m["scenery"]:
        d = {k: v for k, v in o.items() if k != "name"}
        objs.append(d)
    w = _bar_w(a)
    if w > 0:
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                     "visible": True, "w": w, "x": 64 - w, "y": 0})
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                     "visible": True, "w": w, "x": 0, "y": 63})
    for mk in m["markers"]:
        c = "color_11" if mk["active"] else "color_9"
        objs.append({"h": 2, "layer": 1, "tags": [c, "active" if mk["active"] else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": mk["x"], "y": mk["y"]})
    out = []
    if m["mode"] == "block":
        for i, b in enumerate(m["blocks"]):
            out.append({"h": 4, "layer": 1, "name": "block_left" if i == 0 else "block_right",
                        "pixels": [[10] * 4 for _ in range(4)], "tags": ["cyan", "block", "player"],
                        "type": "player", "visible": True, "w": 4, "x": b[0], "y": b[1]})
        start = 2
    else:
        for b in m["blocks"]:
            objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                         "visible": True, "w": 4, "x": b[0], "y": b[1]})
        start = 0
    objs.sort(key=lambda d: (d["y"], d["x"]))
    for r, d in enumerate(objs):
        d["name"] = "%s_%s_%d" % (d["type"], d["tags"][0].split("_")[1], start + r)
        out.append(d)
    return out


def transition_function(state, action):
    m = _parse(state)
    if _mem["canon"] is not None and _canon(state) == _mem["canon"]:
        a = _mem["a"]
    elif m["bar"] == 0:
        a = 0
    else:
        a = next(n for n in range(1000) if _bar_w(n) == m["bar"])
    _step(m, action)
    a += 1
    out = _render(m, a)
    _mem["canon"], _mem["a"] = _canon(out), a
    return out
