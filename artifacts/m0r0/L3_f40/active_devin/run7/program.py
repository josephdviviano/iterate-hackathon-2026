# Mechanics: block mode = two mirrored cyan 4x4 blocks; A1/A2 move both y-4/+4, A3 apart, A4 together; each block moves
# alone if its target cell stays in its board half (x 2..26 / 34..58, y 2..58) and is free of maze/marker cells.
# Click inactive marker -> armed mode (blocks -> color_1 players, marker active 11); armed A1-A4 move the active marker by 4
# (blocked by bounds, maze, other markers, players); click other marker switches, active one no-op, player -> block mode.
# Bars wall_0 grow w=floor(3(a+1)/7) per action (a hidden; continuity-gated). Unconfirmed: maze cells beyond the 3 observed.
import json

MAZE = {(6, 18), (14, 38), (38, 14)}
DELTA = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"canon": None, "a": 0}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(a):
    return (3 * (a + 1)) // 7


def actions_from_state(state):
    w = max([o["w"] for o in state if o["type"] == "wall" and "color_0" in o["tags"]] or [0])
    if w == 0:
        return 0
    a = 0
    while bar_width(a) != w:
        a += 1
    return a


def parse(state):
    m = {"blocks": None, "players": [], "markers": [], "walls": [], "hazards": []}
    blocks = {}
    for o in state:
        if o["name"] in ("block_left", "block_right"):
            blocks[o["name"]] = (o["x"], o["y"])
        elif o["type"] == "player":
            m["players"].append((o["x"], o["y"]))
        elif o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif o["type"] == "wall" and "color_15" in o["tags"]:
            m["walls"].append(o)
        elif o["type"] == "hazard":
            m["hazards"].append(o)
    if blocks:
        m["blocks"] = [list(blocks["block_left"]), list(blocks["block_right"])]
    return m


def marker_cells(m, skip=None):
    return {(mk[0] - 1, mk[1] - 1) for i, mk in enumerate(m["markers"]) if i != skip}


def block_ok(cell, left, m):
    x, y = cell
    lo, hi = (2, 26) if left else (34, 58)
    return lo <= x <= hi and 2 <= y <= 58 and cell not in MAZE and cell not in marker_cells(m)


def move_blocks(m, act):
    dx, dy = DELTA[act]
    for i, b in enumerate(m["blocks"]):
        mdx = dx if dy else (dx if i == 0 else -dx)
        cell = (b[0] + mdx, b[1] + dy)
        if block_ok(cell, i == 0, m):
            b[0], b[1] = cell


def move_marker(m, act):
    dx, dy = DELTA[act]
    for i, mk in enumerate(m["markers"]):
        if not mk[2]:
            continue
        cell = (mk[0] - 1 + dx, mk[1] - 1 + dy)
        blocked = (not (2 <= cell[0] <= 58 and 2 <= cell[1] <= 58) or cell in MAZE
                   or cell in marker_cells(m, skip=i) or cell in {tuple(p) for p in m["players"]})
        if not blocked:
            mk[0], mk[1] = cell[0] + 1, cell[1] + 1


def hit(x, y, ox, oy, w, h):
    return ox <= x < ox + w and oy <= y < oy + h


def click(m, x, y):
    for i, mk in enumerate(m["markers"]):
        if hit(x, y, mk[0], mk[1], 2, 2):
            if m["blocks"] is not None:
                m["players"] = [tuple(b) for b in m["blocks"]]
                m["blocks"] = None
            for j, other in enumerate(m["markers"]):
                other[2] = (j == i)
            return
    if m["blocks"] is None:
        for p in m["players"]:
            if hit(x, y, p[0], p[1], 4, 4):
                m["blocks"] = [list(p) for p in sorted(m["players"], key=lambda p: p[0])]
                m["players"] = []
                for mk in m["markers"]:
                    mk[2] = False
                return


def render(m, a):
    w = bar_width(a)
    items = []  # (y, x, prefix, obj)
    if w > 0:
        items.append((0, 64 - w, "wall_0", {"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                                            "visible": True, "w": w, "x": 64 - w, "y": 0}))
        items.append((63, 0, "wall_0", {"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                                        "visible": True, "w": w, "x": 0, "y": 63}))
    for o in m["walls"]:
        items.append((o["y"], o["x"], "wall_15", {k: v for k, v in o.items() if k != "name"}))
    for o in m["hazards"]:
        items.append((o["y"], o["x"], "hazard_8", {k: v for k, v in o.items() if k != "name"}))
    for x, y, act in m["markers"]:
        c = 11 if act else 9
        items.append((y, x, "marker_%d" % c, {"h": 2, "layer": 1, "tags": ["color_%d" % c, "active" if act else "inactive", "marker"],
                                              "type": "marker", "visible": True, "w": 2, "x": x, "y": y}))
    for x, y in m["players"]:
        items.append((y, x, "player_1", {"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                                         "visible": True, "w": 4, "x": x, "y": y}))
    out = []
    start = 0
    if m["blocks"] is not None:
        start = 2
        for name, (x, y) in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"h": 4, "layer": 1, "name": name, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True, "w": 4, "x": x, "y": y})
    items.sort(key=lambda t: (t[0], t[1]))
    for r, (_, _, prefix, o) in enumerate(items):
        o = dict(o)
        o["name"] = "%s_%d" % (prefix, start + r)
        out.append(o)
    return out


def transition_function(state, action):
    a = _last["a"] if canon(state) == _last["canon"] else actions_from_state(state)
    m = parse(state)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
    elif action in DELTA:
        if m["blocks"] is not None:
            move_blocks(m, action)
        else:
            move_marker(m, action)
    a += 1
    out = render(m, a)
    _last["canon"] = canon(out)
    _last["a"] = a
    return out
