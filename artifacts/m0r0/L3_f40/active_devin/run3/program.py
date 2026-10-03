# Mechanics: mirror-blocks/marker game. Block mode: two cyan 4x4 blocks; A1/A2 move both y-4/+4, A3 apart, A4 together
# (mirrored x), each block stays in its board half and is stopped by markers' cells and hidden maze cells.
# Click inactive marker -> blocks become color_1 armed players, clicked marker active (11); armed A1-A4 move active marker
# by 4 (stopped by bounds, maze, other markers, players); click other marker switches; click player -> blocks mode.
# Bars wall_0 (top-right, bottom-left) have width floor(3(n+1)/7), n = actions since level start. Hypothesis: maze cells.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}   # inferred hidden blocking cells (4x4 cell top-left); unseen cells unknown
LO, HI = 2, 58                          # cell top-left range on the 4-grid
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"state": None, "n": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(n):
    return (3 * (n + 1)) // 7


def parse(state):
    m = {"mode": "block", "blocks": {}, "players": [], "markers": [], "bar": 0, "bg": []}
    for o in state:
        t = o["type"]
        if t == "player" and "block" in o["tags"]:
            m["blocks"][o["name"]] = (o["x"], o["y"])
        elif t == "player":
            m["mode"] = "armed"
            m["players"].append((o["x"], o["y"]))
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif t == "wall" and "color_0" in o["tags"]:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["bg"].append(o)
    return m


def cell_of_marker(mk):
    return (mk[0] - 1, mk[1] - 1)


def in_box(px, py, x, y, w, h):
    return x <= px < x + w and y <= py < y + h


def move_block(name, pos, d, m):
    dx, dy = d
    if name == "block_right":
        dx = -dx
    nx, ny = pos[0] + dx, pos[1] + dy
    lo_x, hi_x = (LO, 26) if name == "block_left" else (34, HI)
    if not (lo_x <= nx <= hi_x and LO <= ny <= HI):
        return pos
    if (nx, ny) in MAZE or any((nx, ny) == cell_of_marker(mk) for mk in m["markers"]):
        return pos
    return (nx, ny)


def move_marker(mk, d, m):
    cx, cy = cell_of_marker(mk)
    nx, ny = cx + d[0], cy + d[1]
    if not (LO <= nx <= HI and LO <= ny <= HI):
        return
    if (nx, ny) in MAZE or (nx, ny) in m["players"]:
        return
    if any(o is not mk and (nx, ny) == cell_of_marker(o) for o in m["markers"]):
        return
    mk[0], mk[1] = nx + 1, ny + 1


def step(m, action):
    if isinstance(action, dict):
        px, py = action["x"], action["y"]
        hit_m = next((mk for mk in m["markers"] if in_box(px, py, mk[0], mk[1], 2, 2)), None)
        if hit_m is not None:
            if not hit_m[2]:
                if m["mode"] == "block":
                    m["players"] = [m["blocks"]["block_left"], m["blocks"]["block_right"]]
                    m["blocks"] = {}
                    m["mode"] = "armed"
                for mk in m["markers"]:
                    mk[2] = mk is hit_m
            return
        if m["mode"] == "armed" and any(in_box(px, py, x, y, 4, 4) for x, y in m["players"]):
            pl = sorted(m["players"])
            m["blocks"] = {"block_left": pl[0], "block_right": pl[-1]}
            m["players"] = []
            m["mode"] = "block"
            for mk in m["markers"]:
                mk[2] = False
        return
    d = MOVES.get(action)
    if d is None:
        return
    if m["mode"] == "block":
        for name in ("block_left", "block_right"):
            m["blocks"][name] = move_block(name, m["blocks"][name], d, m)
    else:
        for mk in m["markers"]:
            if mk[2]:
                move_marker(mk, d, m)


def render(m, n):
    objs = [dict(o) for o in m["bg"]]
    w = bar_width(n)
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"type": "wall", "tags": ["color_0", "wall"], "x": x, "y": y, "w": w, "h": 1,
                         "layer": 0, "visible": True, "_c": "0"})
    for x, y, act in m["markers"]:
        c = "11" if act else "9"
        objs.append({"type": "marker", "tags": ["color_" + c, "active" if act else "inactive", "marker"],
                     "x": x, "y": y, "w": 2, "h": 2, "layer": 1, "visible": True, "_c": c})
    for x, y in m["players"]:
        objs.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": x, "y": y, "w": 4, "h": 4,
                     "layer": 1, "visible": True, "_c": "1"})
    for o in objs:
        if "_c" not in o:
            o["_c"] = next(t[6:] for t in o["tags"] if t.startswith("color_"))
    start = 0
    out = []
    if m["mode"] == "block":
        for name, (x, y) in m["blocks"].items():
            out.append({"name": name, "type": "player", "tags": ["cyan", "block", "player"], "x": x, "y": y,
                        "w": 4, "h": 4, "layer": 1, "visible": True, "pixels": [[10] * 4 for _ in range(4)]})
        start = 2
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(objs):
        c = o.pop("_c")
        o["name"] = "%s_%s_%d" % (o["type"], c, start + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _last["state"] is not None and _canon(state) == _last["state"]:
        n = _last["n"]
    else:
        w = m["bar"]
        n = 0
        if w > 0:
            while bar_width(n) < w:
                n += 1
    step(m, action)
    n += 1
    out = render(m, n)
    _last["state"] = _canon(out)
    _last["n"] = n
    return out
