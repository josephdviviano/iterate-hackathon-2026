# Mechanics: two cyan blocks (block mode) move together: A1/A2 y-/+4, A3 apart, A4 together (mirrored), each blocked by bounds, maze cells and marker cells.
# Clicking a marker arms: blocks become color_1 players, marker becomes active (color_11); A1-A4 then steer the active marker by 4 (blocked by maze/markers/players).
# Clicking another marker switches the active one; clicking a player disarms; clicks on the active marker/block are no-ops. A5 does nothing.
# Timer bars (wall_0, top-right and bottom-left) have width floor(3(n+1)/7), n = actions since level start (hidden, continuity-gated; fallback = smallest n for the width).
# Names = type_color_rank by (y,x) (blocks reserve 0,1). Unconfirmed: invisible maze cells beyond the 3 inferred ones, win/reset conditions.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
STEP = 4
_last = {"out": None, "n": 0}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(n):
    return (3 * (n + 1)) // 7


def parse(state):
    m = {"mode": "block", "blocks": {}, "markers": [], "scenery": [], "bar": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            side = "left" if o["x"] < 32 else "right"
            m["blocks"][side] = [o["x"], o["y"]]
            if "armed" in o["tags"]:
                m["mode"] = "armed"
        elif t == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o["tags"]})
        elif t == "wall" and "color_0" in o["tags"]:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["scenery"].append(dict(o))
    return m


def marker_cell(mk):
    return (mk["x"] - 1, mk["y"] - 1)


def block_free(side, pos, m):
    x, y = pos
    lo, hi = (2, 26) if side == "left" else (34, 58)
    if not (lo <= x <= hi and 2 <= y <= 58):
        return False
    if (x, y) in MAZE:
        return False
    return all(marker_cell(mk) != (x, y) for mk in m["markers"])


def marker_free(mk, cell, m):
    x, y = cell
    if not (2 <= x <= 58 and 2 <= y <= 58):
        return False
    if cell in MAZE:
        return False
    if any(o is not mk and marker_cell(o) == cell for o in m["markers"]):
        return False
    return all(tuple(b) != cell for b in m["blocks"].values())


DELTAS = {1: (0, -STEP), 2: (0, STEP)}


def move_blocks(m, a):
    for side, pos in m["blocks"].items():
        if a in DELTAS:
            dx, dy = DELTAS[a]
        else:
            outward = -STEP if side == "left" else STEP
            dx, dy = (outward if a == 3 else -outward), 0
        new = [pos[0] + dx, pos[1] + dy]
        if block_free(side, new, m):
            m["blocks"][side] = new


def move_marker(m, a):
    act = [mk for mk in m["markers"] if mk["active"]]
    if not act:
        return
    mk = act[0]
    dx, dy = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}[a]
    cx, cy = marker_cell(mk)
    if marker_free(mk, (cx + dx, cy + dy), m):
        mk["x"] += dx
        mk["y"] += dy


def hit(px, py, x, y, w, h):
    return x <= px < x + w and y <= py < y + h


def click(m, px, py):
    for mk in m["markers"]:
        if hit(px, py, mk["x"], mk["y"], 2, 2):
            if mk["active"]:
                return
            for o in m["markers"]:
                o["active"] = o is mk
            m["mode"] = "armed"
            return
    for pos in m["blocks"].values():
        if hit(px, py, pos[0], pos[1], 4, 4):
            if m["mode"] == "armed":
                m["mode"] = "block"
                for o in m["markers"]:
                    o["active"] = False
            return


def render(m, n):
    objs = [dict(o) for o in m["scenery"]]
    w = bar_width(n)
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": w, "x": x, "y": y, "_p": "wall_0"})
    for o in objs:
        if "_p" not in o:
            o["_p"] = o["type"] + "_" + o["tags"][0].split("_")[1]
    for mk in m["markers"]:
        col = 11 if mk["active"] else 9
        objs.append({"h": 2, "layer": 1, "tags": ["color_%d" % col, "active" if mk["active"] else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": mk["x"], "y": mk["y"], "_p": "marker_%d" % col})
    fixed = []
    for side in ("left", "right"):
        if side not in m["blocks"]:
            continue
        x, y = m["blocks"][side]
        base = {"h": 4, "layer": 1, "type": "player", "visible": True, "w": 4, "x": x, "y": y}
        if m["mode"] == "block":
            base.update(name="block_" + side, tags=["cyan", "block", "player"],
                        pixels=[[10] * 4 for _ in range(4)])
            fixed.append(base)
        else:
            base.update(tags=["color_1", "armed", "player"], _p="player_1")
            objs.append(base)
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = len(fixed)
    for i, o in enumerate(objs):
        o["name"] = "%s_%d" % (o.pop("_p"), start + i)
    return fixed + objs


def transition_function(state, action):
    m = parse(state)
    if _last["out"] is not None and canon(state) == _last["out"]:
        n = _last["n"]
    else:
        n = 0
        while bar_width(n) < m["bar"]:
            n += 1
    if isinstance(action, dict):
        click(m, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if m["mode"] == "block":
            move_blocks(m, action)
        else:
            move_marker(m, action)
    n += 1
    out = render(m, n)
    _last["out"], _last["n"] = canon(out), n
    return out
