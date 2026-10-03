# Mechanics: two mirrored cyan blocks (block mode) move y-/+4 on A1/A2; A3 = apart (left x-4, right x+4), A4 = together;
# each block moves alone and is blocked by bounds, a hidden maze cell or a marker's 4x4 cell (marker xy - 1).
# A6 click on an inactive marker arms it (blocks -> color_1 'armed' players, marker -> color_11 active); in armed mode
# A1-A4 move the active marker by 4 (same blocking); clicking a player returns to block mode; other clicks / A5 no-ops.
# Bars color_0 at top-right/bottom-left have width floor(3(a+1)/7), a = actions taken; names = type_color_rank by (y,x).
# Unconfirmed: maze cells beyond the 3 inferred, marker-vs-player blocking, A7, bounds edges.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
STEP = 4
_memo = {"out": None, "a": 0}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def bar_width(a):
    return (3 * (a + 1)) // 7


def actions_from_bar(w):
    return 0 if w <= 0 else (7 * w - 1) // 3


def parse(state):
    m = {"mode": "block", "blocks": {}, "players": [], "markers": [], "static": [], "bar": 0}
    for o in state:
        t, tags = o["type"], o.get("tags", [])
        if t == "player" and "block" in tags:
            side = "left" if o["name"] == "block_left" else "right"
            m["blocks"][side] = [o["x"], o["y"]]
        elif t == "player":
            m["mode"] = "armed"
            m["players"].append([o["x"], o["y"]])
        elif t == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in tags})
        elif "color_0" in tags:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["static"].append(dict(o))
    if m["mode"] == "armed":
        ps = sorted(m["players"], key=lambda p: p[0])
        m["blocks"] = {"left": ps[0], "right": ps[-1]}
    return m


def in_bounds(x, y):
    return 0 <= x and x + 4 <= 64 and 1 <= y and y + 4 <= 63


def marker_cell(mk):
    return (mk["x"] - 1, mk["y"] - 1)


def free_cell(x, y, occupied):
    return in_bounds(x, y) and (x, y) not in MAZE and (x, y) not in occupied


def move_blocks(m, action):
    occ = {marker_cell(mk) for mk in m["markers"]}
    deltas = {1: {"left": (0, -STEP), "right": (0, -STEP)},
              2: {"left": (0, STEP), "right": (0, STEP)},
              3: {"left": (-STEP, 0), "right": (STEP, 0)},
              4: {"left": (STEP, 0), "right": (-STEP, 0)}}[action]
    for side, (dx, dy) in deltas.items():
        x, y = m["blocks"][side]
        if free_cell(x + dx, y + dy, occ):
            m["blocks"][side] = [x + dx, y + dy]


def move_marker(m, action):
    dx, dy = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}[action]
    act = [mk for mk in m["markers"] if mk["active"]]
    if not act:
        return
    mk = act[0]
    occ = {marker_cell(o) for o in m["markers"] if o is not mk}
    occ |= {tuple(p) for p in m["blocks"].values()}
    cx, cy = marker_cell(mk)
    if free_cell(cx + dx, cy + dy, occ):
        mk["x"] += dx
        mk["y"] += dy


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def click(m, x, y):
    for mk in m["markers"]:
        if hit(x, y, mk["x"], mk["y"], 2):
            if mk["active"]:
                return
            for o in m["markers"]:
                o["active"] = o is mk
            m["mode"] = "armed"
            return
    if m["mode"] == "armed":
        for bx, by in m["blocks"].values():
            if hit(x, y, bx, by, 4):
                m["mode"] = "block"
                for o in m["markers"]:
                    o["active"] = False
                return


def step(m, action):
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
        return
    if action in (1, 2, 3, 4):
        if m["mode"] == "block":
            move_blocks(m, action)
        else:
            move_marker(m, action)


def render(m, a):
    objs = []
    for o in m["static"]:
        objs.append((o["name"].rsplit("_", 1)[0], o))
    for mk in m["markers"]:
        col = "11" if mk["active"] else "9"
        objs.append(("marker_" + col, {"h": 2, "layer": 1, "tags": ["color_" + col, "active" if mk["active"] else "inactive", "marker"],
                                        "type": "marker", "visible": True, "w": 2, "x": mk["x"], "y": mk["y"]}))
    w = bar_width(a)
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append(("wall_0", {"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                                    "visible": True, "w": w, "x": bx, "y": by}))
    out = []
    if m["mode"] == "block":
        for side in ("left", "right"):
            bx, by = m["blocks"][side]
            out.append({"h": 4, "layer": 1, "name": "block_" + side, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True, "w": 4, "x": bx, "y": by})
        start = 2
    else:
        for bx, by in m["blocks"].values():
            objs.append(("player_1", {"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                                      "visible": True, "w": 4, "x": bx, "y": by}))
        start = 0
    objs.sort(key=lambda po: (po[1]["y"], po[1]["x"]))
    for i, (prefix, o) in enumerate(objs):
        o = dict(o)
        o["name"] = "%s_%d" % (prefix, start + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _memo["out"] is not None and canon(state) == _memo["out"]:
        a = _memo["a"]
    else:
        a = actions_from_bar(m["bar"])
    step(m, action)
    a += 1
    out = render(m, a)
    _memo["out"], _memo["a"] = canon(out), a
    return out
