# Mechanics: two modes. BLOCK mode: cyan block_left/right move 4px per arrow, mirrored in x (3: L-4/R+4,
# 4: L+4/R-4, 1/2: both y-/+4), each blocked by markers / its half / board. Click a marker -> ARMED mode:
# players become player_1 (armed), clicked marker active (color 11); arrows move only the active marker 4px;
# click another marker switches; click a player -> back to BLOCK mode. Step bar wall_0 (top-right, bottom-left)
# width floor(3(c+1)/7), c = hidden step count. Unconfirmed: 5 blocked moves (t3-5,11,18) imply an invisible maze.
import copy

BLOCK_PIX = [[10] * 4 for _ in range(4)]
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"out": None, "c": 0}


def bar_width(c):
    return min(64, (3 * (c + 1)) // 7)


def overlaps(a, b):
    return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
            and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])


def hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def parse(state):
    m = {"players": [], "markers": [], "bg": [], "bar": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            m["players"].append({"x": o["x"], "y": o["y"], "w": 4, "h": 4})
            m["armed"] = "armed" in o["tags"]
        elif t == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"],
                                 "active": "active" in o["tags"]})
        elif t == "wall" and "color_0" in o["tags"]:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["bg"].append(o)
    m.setdefault("armed", False)
    return m


def blocked_player(p, nx, ny, m):
    half_left = p["x"] < 32
    if half_left and (nx < 0 or nx + 4 > 32):
        return True
    if not half_left and (nx < 32 or nx + 4 > 64):
        return True
    if ny < 1 or ny + 4 > 63:
        return True
    tgt = {"x": nx, "y": ny, "w": 4, "h": 4}
    return any(overlaps(tgt, mk) for mk in m["markers"])


def blocked_marker(mk, nx, ny, m):
    if nx < 0 or ny < 1 or nx + mk["w"] > 64 or ny + mk["h"] > 63:
        return True
    tgt = {"x": nx, "y": ny, "w": mk["w"], "h": mk["h"]}
    return any(o is not mk and overlaps(tgt, o) for o in m["markers"])


def step_players(m, a):
    dx, dy = DIRS[a]
    for p in m["players"]:
        sx = dx if p["x"] < 32 else -dx
        nx, ny = p["x"] + sx, p["y"] + dy
        if not blocked_player(p, nx, ny, m):
            p["x"], p["y"] = nx, ny


def step_marker(m, a):
    dx, dy = DIRS[a]
    for mk in m["markers"]:
        if mk["active"]:
            nx, ny = mk["x"] + dx, mk["y"] + dy
            if not blocked_marker(mk, nx, ny, m):
                mk["x"], mk["y"] = nx, ny


def click(m, x, y):
    for mk in m["markers"]:
        if hit(mk, x, y):
            if not mk["active"]:
                for o in m["markers"]:
                    o["active"] = False
                mk["active"] = True
                m["armed"] = True
            return
    if m["armed"] and any(hit(p, x, y) for p in m["players"]):
        m["armed"] = False
        for o in m["markers"]:
            o["active"] = False


def render(m, w):
    objs = []
    for o in m["bg"]:
        objs.append({k: v for k, v in o.items() if k != "name"})
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append({"type": "wall", "tags": ["color_0", "wall"], "x": bx, "y": by,
                         "w": w, "h": 1, "layer": 0, "visible": True, "_c": 0})
    for mk in m["markers"]:
        col = 11 if mk["active"] else 9
        objs.append({"type": "marker", "tags": ["color_%d" % col, "active" if mk["active"] else "inactive",
                                                "marker"],
                     "x": mk["x"], "y": mk["y"], "w": mk["w"], "h": mk["h"], "layer": 1,
                     "visible": True, "_c": col})
    out, idx = [], 0
    if m["armed"]:
        for p in m["players"]:
            objs.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": p["x"],
                         "y": p["y"], "w": 4, "h": 4, "layer": 1, "visible": True, "_c": 1})
    else:
        for p in sorted(m["players"], key=lambda p: p["x"]):
            out.append({"name": "block_left" if p["x"] < 32 else "block_right", "type": "player",
                        "tags": ["cyan", "block", "player"], "x": p["x"], "y": p["y"], "w": 4, "h": 4,
                        "layer": 1, "visible": True, "pixels": copy.deepcopy(BLOCK_PIX)})
        idx = len(out)
    for o in sorted(objs, key=lambda o: (o["y"], o["x"])):
        col = o.pop("_c", None)
        if col is None:
            col = int([t for t in o["tags"] if t.startswith("color_")][0][6:])
        o["name"] = "%s_%d_%d" % (o["type"], col, idx)
        idx += 1
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if canon(state) == _mem["out"] or (_mem["out"] is not None and bar_width(_mem["c"]) == m["bar"] > 0):
        c = _mem["c"]
    else:
        c = 0
        while bar_width(c) < m["bar"]:
            c += 1
    c += 1
    aid = action["action_id"] if isinstance(action, dict) else action
    if aid in DIRS:
        if m["armed"]:
            step_marker(m, aid)
        else:
            step_players(m, aid)
    elif aid == 6:
        click(m, action["x"], action["y"])
    out = render(m, bar_width(c))
    _mem["out"], _mem["c"] = canon(out), c
    return out
