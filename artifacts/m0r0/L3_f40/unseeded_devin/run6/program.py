# Mechanics: block mode: A1/A2 move both cyan blocks y-4/y+4, A3 apart (left x-4,right x+4), A4 together.
# Clicking a marker arms (blocks -> color_1 players, marker active); A1-4 then move the active marker by 4;
# clicking another marker switches it, clicking a player disarms. Moves blocked by bounds, marker/player cells
# and an invisible maze (cells inferred from blocked moves; unseen maze cells are an unconfirmed hypothesis).
# Timer bars w=floor(3(n+1)/7), n=actions incl. current (continuity-gated); names=type_color_rank by (y,x).
import json

MAZE = {(6, 18), (14, 38), (38, 14)}  # 4x4 cell top-left coords inferred from blocked moves
LO, HI, MID = 2, 58, 32
_last = {"canon": None, "n": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(n):
    return (3 * (n + 1)) // 7


def parse(state):
    m = {"blocks": None, "players": [], "markers": [], "scenery": []}
    blocks = {}
    for o in state:
        t = o["type"]
        if o["name"] in ("block_left", "block_right"):
            blocks[o["name"]] = (o["x"], o["y"])
        elif t == "player":
            m["players"].append((o["x"], o["y"]))
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif t == "wall" and "color_0" in o["tags"]:
            m["bar"] = max(m.get("bar", 0), o["w"])
        else:
            m["scenery"].append(o)
    if blocks:
        m["blocks"] = blocks
    return m


def cell_free(cell, occupied):
    return cell not in MAZE and cell not in occupied


def in_half(x, left):
    return (LO <= x <= MID - 6) if left else (MID + 2 <= x <= HI)


def move_blocks(m, action):
    dy = {1: -4, 2: 4}.get(action, 0)
    dx = {3: -4, 4: 4}.get(action, 0)
    if not dx and not dy:
        return
    occ = {(mx - 1, my - 1) for mx, my, _ in m["markers"]}
    for name, sgn in (("block_left", 1), ("block_right", -1)):
        x, y = m["blocks"][name]
        nx, ny = x + dx * sgn, y + dy
        if in_half(nx, name == "block_left") and LO <= ny <= HI and cell_free((nx, ny), occ):
            m["blocks"][name] = (nx, ny)


def move_marker(m, action):
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
    if not d:
        return
    act = [mk for mk in m["markers"] if mk[2]]
    if not act:
        return
    mk = act[0]
    occ = {(x - 1, y - 1) for x, y, a in m["markers"] if not a} | set(m["players"])
    nx, ny = mk[0] + d[0], mk[1] + d[1]
    if LO <= nx - 1 <= HI and LO <= ny - 1 <= HI and cell_free((nx - 1, ny - 1), occ):
        mk[0], mk[1] = nx, ny


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def click(m, x, y):
    for mk in m["markers"]:
        if hit(x, y, mk[0], mk[1], 2):
            if m["blocks"] is not None:
                m["players"] = list(m["blocks"].values())
                m["blocks"] = None
            for other in m["markers"]:
                other[2] = other is mk
            return
    if m["blocks"] is None:
        for px, py in m["players"]:
            if hit(x, y, px, py, 4):
                m["blocks"] = {}
                for qx, qy in m["players"]:
                    m["blocks"]["block_left" if qx < MID else "block_right"] = (qx, qy)
                m["players"] = []
                for mk in m["markers"]:
                    mk[2] = False
                return


def obj(typ, color, tags, x, y, w, h, layer):
    return {"type": typ, "tags": [f"color_{color}"] + tags, "x": x, "y": y, "w": w, "h": h,
            "layer": layer, "visible": True, "_c": color}


def render(m, n):
    ranked = []
    for o in m["scenery"]:
        o = dict(o)
        o["_c"] = o["tags"][0].split("_", 1)[1]
        ranked.append(o)
    for x, y, a in m["markers"]:
        ranked.append(obj("marker", 11 if a else 9, ["active" if a else "inactive", "marker"],
                          x, y, 2, 2, 1))
    for x, y in m["players"]:
        ranked.append(obj("player", 1, ["armed", "player"], x, y, 4, 4, 1))
    w = bar_width(n)
    if w > 0:
        ranked.append(obj("wall", 0, ["wall"], 64 - w, 0, w, 1, 0))
        ranked.append(obj("wall", 0, ["wall"], 0, 63, w, 1, 0))
    out = []
    base = 0
    if m["blocks"] is not None:
        base = 2
        for name in ("block_left", "block_right"):
            x, y = m["blocks"][name]
            out.append({"name": name, "type": "player", "tags": ["cyan", "block", "player"],
                        "x": x, "y": y, "w": 4, "h": 4, "layer": 1, "visible": True,
                        "pixels": [[10] * 4 for _ in range(4)]})
    ranked.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(ranked):
        c = o.pop("_c")
        o["name"] = f"{o['type']}_{c}_{base + i}"
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _last["canon"] is not None and _canon(state) == _last["canon"]:
        n = _last["n"]
    else:
        w = m.get("bar", 0)
        n = (7 * w - 1) // 3 if w > 0 else 0
    n += 1
    if isinstance(action, dict):
        click(m, action["x"], action["y"])
    elif m["blocks"] is not None:
        move_blocks(m, action)
    else:
        move_marker(m, action)
    out = render(m, n)
    _last["canon"], _last["n"] = _canon(out), n
    return out
