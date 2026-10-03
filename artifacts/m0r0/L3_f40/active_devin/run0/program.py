# Mechanics: two mirrored cyan blocks (block mode) move together: A1/A2 up/down 4, A3 apart, A4 together;
# each block is stopped by markers, the board half it lives in, and an invisible maze (hidden layout cells).
# Click inactive marker -> blocks become armed color_1 players, that marker turns active (color 11);
# clicking the active marker does nothing; clicking a player -> back to blocks. Armed: A1-A4 move the active marker.
# Bars wall_0 grow as floor(3m/7), m = actions since level start + 1 (hidden; fallback smallest m). Maze cells are inferred.
import json

STEP = 4
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
BLOCKED_CELLS = {(14, 38), (6, 18), (38, 14)}  # inferred invisible maze cells (4x4 top-left)
CYAN = [[10] * 4 for _ in range(4)]
_hidden = {"state": None, "m": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def bar_width(m):
    return (3 * m) // 7


def fallback_m(w):
    m = 1
    while bar_width(m) < w:
        m += 1
    return m


def in_maze(rect):
    return any(overlap(rect, (cx, cy, 4, 4)) for cx, cy in BLOCKED_CELLS)


def hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def parse(state):
    model = {"blocks": [], "players": [], "markers": [], "static": [], "bar": 0}
    for o in state:
        tags = o.get("tags", [])
        if "block" in tags:
            model["blocks"].append(dict(o))
        elif "armed" in tags:
            model["players"].append(dict(o))
        elif o["type"] == "marker":
            model["markers"].append(dict(o))
        elif "color_0" in tags:
            if o["y"] == 0:
                model["bar"] = o["w"]
        else:
            model["static"].append(dict(o))
    return model


def block_move_ok(b, dx, dy, markers, left_half):
    r = (b["x"] + dx, b["y"] + dy, b["w"], b["h"])
    lo, hi = (0, 32) if left_half else (32, 64)
    if r[0] < lo or r[0] + r[2] > hi or r[1] < 1 or r[1] + r[3] > 63:
        return False
    if any(overlap(r, (m["x"], m["y"], m["w"], m["h"])) for m in markers):
        return False
    return not in_maze(r)


def marker_move_ok(mk, dx, dy, others):
    r = (mk["x"] + dx, mk["y"] + dy, mk["w"], mk["h"])
    if r[0] < 0 or r[0] + r[2] > 64 or r[1] < 1 or r[1] + r[3] > 63:
        return False
    if any(overlap(r, (o["x"], o["y"], o["w"], o["h"])) for o in others):
        return False
    return not in_maze(r)


def set_marker(mk, active):
    c = 11 if active else 9
    mk["tags"] = ["color_%d" % c, "active" if active else "inactive", "marker"]
    mk["_color"] = c


def to_players(blocks):
    out = []
    for b in blocks:
        out.append({"h": b["h"], "layer": b["layer"], "tags": ["color_1", "armed", "player"],
                    "type": "player", "visible": True, "w": b["w"], "x": b["x"], "y": b["y"]})
    return out


def to_blocks(players):
    ps = sorted(players, key=lambda p: p["x"])
    out = []
    for p, nm in zip(ps, ["block_left", "block_right"]):
        out.append({"h": 4, "layer": 1, "name": nm, "pixels": [r[:] for r in CYAN],
                    "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                    "w": 4, "x": p["x"], "y": p["y"]})
    return out


def step_blocks(model, action):
    if action in (1, 2):
        dx, dy = MOVES[action]
        moves = {"block_left": (dx, dy), "block_right": (dx, dy)}
    elif action == 3:
        moves = {"block_left": (-STEP, 0), "block_right": (STEP, 0)}
    elif action == 4:
        moves = {"block_left": (STEP, 0), "block_right": (-STEP, 0)}
    else:
        return
    for b in model["blocks"]:
        dx, dy = moves[b["name"]]
        if block_move_ok(b, dx, dy, model["markers"], b["name"] == "block_left"):
            b["x"] += dx
            b["y"] += dy


def step_active_marker(model, action):
    if action not in MOVES:
        return
    dx, dy = MOVES[action]
    for mk in model["markers"]:
        if "active" in mk["tags"]:
            others = [m for m in model["markers"] if m is not mk] + model["players"]
            if marker_move_ok(mk, dx, dy, others):
                mk["x"] += dx
                mk["y"] += dy


def click(model, x, y):
    armed = bool(model["players"])
    target = next((m for m in model["markers"] if hit(m, x, y)), None)
    if target is not None:
        if "active" in target["tags"]:
            return
        for m in model["markers"]:
            set_marker(m, m is target)
        if not armed:
            model["players"] = to_players(model["blocks"])
            model["blocks"] = []
        return
    if armed and any(hit(p, x, y) for p in model["players"]):
        model["blocks"] = to_blocks(model["players"])
        model["players"] = []
        for m in model["markers"]:
            set_marker(m, False)


def render(model, m):
    w = bar_width(m)
    others = model["static"] + model["markers"] + model["players"]
    if w > 0:
        base = {"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True, "w": w}
        others.append(dict(base, x=64 - w, y=0, _prefix="wall_0"))
        others.append(dict(base, x=0, y=63, _prefix="wall_0"))
    out = [dict(b) for b in sorted(model["blocks"], key=lambda b: b["name"])]
    idx = len(out)
    for o in sorted(others, key=lambda o: (o["y"], o["x"])):
        o = dict(o)
        if "_prefix" in o:
            prefix = o.pop("_prefix")
        elif o["type"] == "marker":
            prefix = "marker_%d" % (11 if "active" in o["tags"] else 9)
        elif "armed" in o.get("tags", []):
            prefix = "player_1"
        else:
            prefix = o["name"].rsplit("_", 1)[0]
        o.pop("_color", None)
        o["name"] = "%s_%d" % (prefix, idx)
        idx += 1
        out.append(o)
    return out


def transition_function(state, action):
    model = parse(state)
    if _hidden["state"] is not None and canon(state) == _hidden["state"]:
        m = _hidden["m"]
    else:
        m = fallback_m(model["bar"])
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(model, action["x"], action["y"])
    elif model["players"]:
        step_active_marker(model, action)
    else:
        step_blocks(model, action)
    out = render(model, m + 1)
    _hidden["state"] = canon(out)
    _hidden["m"] = m + 1
    return out
