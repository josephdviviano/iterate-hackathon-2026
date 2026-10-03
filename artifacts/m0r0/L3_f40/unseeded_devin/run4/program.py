# Mechanics: two cyan blocks (left half wall_15, right half hazard_8) move together: A1/A2 y-/+4, A3 apart, A4 together;
# each block is blocked by its half's bounds, hidden maze cells and marker cells. Clicking an inactive marker arms
# the blocks (color_1 players) and activates that marker (color_11); A1-A4 then move the active marker by 4. Clicking a
# player disarms back to blocks. Timer bars (color_0) at top-right and bottom-left have w=floor(3(n+1)/7), n=actions.
# Unconfirmed: maze cells beyond {(6,18),(14,38),(38,14)} unseen; n is ambiguous from bar width (continuity-tracked).
import json

MAZE = {(6, 18), (14, 38), (38, 14)}
STEP = 4
_last = {"canon": None, "n": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def is_bar(o):
    return o["type"] == "wall" and color_of(o) == 0


def bar_width(n):
    return (3 * (n + 1)) // 7


def parse(state):
    m = {"static": [], "markers": [], "blocks": [], "armed": False, "w": 0}
    for o in state:
        if is_bar(o):
            m["w"] = max(m["w"], o["w"])
        elif o["type"] == "player":
            m["blocks"].append([o["x"], o["y"]])
            if "armed" in o["tags"]:
                m["armed"] = True
        elif o["type"] == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o["tags"]})
        else:
            m["static"].append(dict(o))
    m["blocks"].sort()  # [left, right]
    return m


def cell_of_marker(mk):
    return (mk["x"] - 1, mk["y"] - 1)


def in_bounds_block(i, x, y):
    lo, hi = (2, 26) if i == 0 else (34, 58)
    return lo <= x <= hi and 2 <= y <= 58


def blocked(cell, occupied):
    return cell in MAZE or cell in occupied


def move_blocks(m, action):
    dy = {1: -STEP, 2: STEP}.get(action, 0)
    dxs = {3: (-STEP, STEP), 4: (STEP, -STEP)}.get(action, (0, 0))
    occ = {cell_of_marker(mk) for mk in m["markers"]}
    for i, b in enumerate(m["blocks"]):
        nx, ny = b[0] + dxs[i], b[1] + dy
        if in_bounds_block(i, nx, ny) and not blocked((nx, ny), occ):
            b[0], b[1] = nx, ny


def move_marker(m, action):
    d = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}[action]
    act = [mk for mk in m["markers"] if mk["active"]]
    if not act:
        return
    a = act[0]
    occ = {cell_of_marker(mk) for mk in m["markers"] if mk is not a}
    occ |= {tuple(b) for b in m["blocks"]}
    nc = (a["x"] - 1 + d[0], a["y"] - 1 + d[1])
    if 2 <= nc[0] <= 58 and 2 <= nc[1] <= 58 and not blocked(nc, occ):
        a["x"], a["y"] = nc[0] + 1, nc[1] + 1


def hit(x, y, ox, oy, w, h):
    return ox <= x < ox + w and oy <= y < oy + h


def click(m, x, y):
    for b in m["blocks"]:
        if hit(x, y, b[0], b[1], 4, 4):
            if m["armed"]:
                m["armed"] = False
                for mk in m["markers"]:
                    mk["active"] = False
            return
    for mk in m["markers"]:
        if hit(x, y, mk["x"], mk["y"], 2, 2):
            if not mk["active"]:
                for o in m["markers"]:
                    o["active"] = False
                mk["active"] = True
                m["armed"] = True
            return


def render(m, n):
    objs = [dict(o) for o in m["static"]]
    for mk in m["markers"]:
        c = 11 if mk["active"] else 9
        objs.append({"h": 2, "w": 2, "layer": 1, "type": "marker", "visible": True, "x": mk["x"], "y": mk["y"],
                     "tags": ["color_%d" % c, "active" if mk["active"] else "inactive", "marker"]})
    w = bar_width(n)
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "w": w, "layer": 0, "type": "wall", "visible": True, "x": bx, "y": by,
                         "tags": ["color_0", "wall"]})
    if m["armed"]:
        for b in m["blocks"]:
            objs.append({"h": 4, "w": 4, "layer": 1, "type": "player", "visible": True, "x": b[0], "y": b[1],
                         "tags": ["color_1", "armed", "player"]})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = 0 if m["armed"] else 2
    for i, o in enumerate(objs):
        o["name"] = "%s_%d_%d" % (o["type"], color_of(o), start + i)
    if not m["armed"]:
        for name, b in zip(("block_left", "block_right"), m["blocks"]):
            objs.append({"h": 4, "w": 4, "layer": 1, "type": "player", "visible": True, "x": b[0], "y": b[1],
                         "name": name, "tags": ["cyan", "block", "player"], "pixels": [[10] * 4 for _ in range(4)]})
    return objs


def transition_function(state, action):
    m = parse(state)
    if _last["canon"] is not None and canon(state) == _last["canon"]:
        n = _last["n"]
    else:
        n = (7 * m["w"] - 1) // 3 if m["w"] > 0 else 0
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if m["armed"]:
            move_marker(m, action)
        else:
            move_blocks(m, action)
    n += 1
    out = render(m, n)
    _last["canon"], _last["n"] = canon(out), n
    return out
