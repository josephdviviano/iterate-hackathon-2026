# Mechanics: two modes. Block mode: cyan block_left/right; ACTION1/2 move both up/down 4, ACTION3/4 move them apart/together
# (mirrored); clicking a marker arms: blocks -> color_1 'armed' players, clicked marker active (11). Armed mode: ACTION1-4 move
# the active marker by 4; clicking another marker switches; clicking a player returns to blocks. Moves into an obstacle cell
# (markers, players, half boundary, or the invisible maze HIDDEN_WALLS inferred from failed moves) are skipped. Bars wall_0 at
# top-right/bottom-left have width floor(3(n+1)/7), n = actions since level start (hidden; maze layout unconfirmed beyond data).
import json

HIDDEN_WALLS = {(14, 38), (6, 18), (38, 14)}  # 4x4 cells (top-left) of the invisible maze, unconfirmed layout
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"last": None, "n": 0, "bar": 0}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def bar_width(n):
    return 3 * (n + 1) // 7


def parse(state):
    g = {"blocks": {}, "players": [], "markers": [], "active": None, "static": [], "bar": 0}
    for o in state:
        if o["name"] in ("block_left", "block_right"):
            g["blocks"][o["name"]] = (o["x"], o["y"])
        elif o["type"] == "player":
            g["players"].append((o["x"], o["y"]))
        elif o["type"] == "marker":
            g["markers"].append((o["x"], o["y"]))
            if "active" in o["tags"]:
                g["active"] = (o["x"], o["y"])
        elif o["type"] == "wall" and color_of(o) == 0:
            if o["y"] == 0:
                g["bar"] = o["w"]
        else:
            g["static"].append(o)
    return g


def halves(g):
    res = {}
    for o in g["static"]:
        res["left" if o["x"] < 32 else "right"] = (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])
    return res


def cell_free_for_block(g, x, y, side):
    box = halves(g).get(side)
    if box and not (box[0] <= x and x + 4 <= box[2] and box[1] <= y and y + 4 <= box[3]):
        return False
    if (x, y) in HIDDEN_WALLS:
        return False
    return not any(x <= mx < x + 4 and y <= my < y + 4 for mx, my in g["markers"])


def cell_free_for_marker(g, x, y):
    if not (0 <= x and x + 2 <= 64 and 1 <= y and y + 2 <= 63):
        return False
    if (x - 1, y - 1) in HIDDEN_WALLS:
        return False
    if any((mx, my) == (x, y) for mx, my in g["markers"]):
        return False
    return not any(px <= x < px + 4 and py <= y < py + 4 for px, py in g["players"])


def step(g, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    if g["blocks"]:
        if aid in DIRS:
            dx, dy = DIRS[aid]
            for name, sgn, side in (("block_left", 1, "left"), ("block_right", -1, "right")):
                if name in g["blocks"]:
                    x, y = g["blocks"][name]
                    nx, ny = x + (dx if sgn == 1 else -dx), y + dy
                    if cell_free_for_block(g, nx, ny, side):
                        g["blocks"][name] = (nx, ny)
        elif aid == 6:
            cx, cy = action["x"], action["y"]
            for mx, my in g["markers"]:
                if mx <= cx < mx + 2 and my <= cy < my + 2:
                    g["players"] = list(g["blocks"].values())
                    g["blocks"] = {}
                    g["active"] = (mx, my)
                    break
    else:
        if aid in DIRS and g["active"] is not None:
            dx, dy = DIRS[aid]
            ax, ay = g["active"]
            nx, ny = ax + dx, ay + dy
            if cell_free_for_marker(g, nx, ny):
                g["markers"] = [(nx, ny) if m == (ax, ay) else m for m in g["markers"]]
                g["active"] = (nx, ny)
        elif aid == 6:
            cx, cy = action["x"], action["y"]
            for mx, my in g["markers"]:
                if mx <= cx < mx + 2 and my <= cy < my + 2:
                    g["active"] = (mx, my)
                    return
            for px, py in g["players"]:
                if px <= cx < px + 4 and py <= cy < py + 4:
                    for p in g["players"]:
                        g["blocks"]["block_left" if p[0] < 32 else "block_right"] = p
                    g["players"] = []
                    g["active"] = None
                    return


def render(g):
    out, others = [], []
    for name in ("block_left", "block_right"):
        if name in g["blocks"]:
            x, y = g["blocks"][name]
            out.append({"h": 4, "layer": 1, "name": name, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True, "w": 4, "x": x, "y": y})
    for o in g["static"]:
        o = dict(o)
        o.pop("name")
        others.append(o)
    for x, y in g["players"]:
        others.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                       "visible": True, "w": 4, "x": x, "y": y})
    for m in g["markers"]:
        act = m == g["active"]
        others.append({"h": 2, "layer": 1, "tags": ["color_11" if act else "color_9", "active" if act else "inactive",
                                                    "marker"], "type": "marker", "visible": True, "w": 2, "x": m[0], "y": m[1]})
    w = g["bar"]
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            others.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                           "w": w, "x": x, "y": y})
    others.sort(key=lambda o: (o["y"], o["x"]))
    base = 2 if g["blocks"] else 0
    for i, o in enumerate(others):
        o["name"] = "%s_%d_%d" % (o["type"], color_of(o), base + i)
        out.append(o)
    return out


def transition_function(state, action):
    g = parse(state)
    if _mem["last"] == canon(state):
        n = _mem["n"]
    else:
        n = 0
        while bar_width(n) != g["bar"]:
            n += 1
    step(g, action)
    n += 1
    g["bar"] = bar_width(n)
    out = render(g)
    _mem.update(last=canon(out), n=n, bar=g["bar"])
    return out
