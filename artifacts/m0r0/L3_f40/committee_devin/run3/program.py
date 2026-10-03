# Mechanics: blocks mode = two cyan 4x4 blocks; ACTION1/2 move both up/down 4, ACTION3/4 move them apart/together (mirrored x).
# Clicking an inactive marker arms: blocks -> color_1 'armed' players, clicked marker active (11), others inactive (9);
# in armed mode ACTION1-4 move the active marker by 4; clicking a player disarms; other clicks / ACTION5 do nothing.
# Movers are blocked by other markers/players and by an invisible maze (no pixels in schema): blocked 4x4 cells are a layout constant.
# color_0 bars at top-right/bottom-left have width floor(3(n+1)/7), n = actions since level start (hidden; fallback = smallest n).
import json

CELL = 4
MAZE = {(14, 38), (6, 18), (38, 14)}  # inferred blocked 4x4 cells (top-left); hypothesis, not visible in state
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_hidden = {"last": None, "n": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(n):
    return (3 * (n + 1)) // 7


def smallest_n(w):
    n = 0
    while bar_width(n) < w:
        n += 1
    return n


def cell_of(x, y, size):
    off = (CELL - size) // 2
    return (x - off, y - off)


def cell_blocked(cx, cy, others):
    if (cx, cy) in MAZE:
        return True
    if cx < 0 or cy < 1 or cx + CELL > 64 or cy + CELL > 63:
        return True
    return (cx, cy) in others


def parse(state):
    m = {"blocks": [], "players": [], "markers": [], "bar": 0, "static": []}
    for o in state:
        t, tags = o["type"], o.get("tags", [])
        if t == "player" and "block" in tags:
            m["blocks"].append([o["x"], o["y"]])
        elif t == "player":
            m["players"].append([o["x"], o["y"]])
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in tags])
        elif t == "wall" and "color_0" in tags:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["static"].append(o)
    return m



def click(m, x, y):
    armed = not m["blocks"]
    for mk in m["markers"]:
        if mk[0] <= x < mk[0] + 2 and mk[1] <= y < mk[1] + 2:
            if mk[2]:
                return
            for other in m["markers"]:
                other[2] = other is mk
            if not armed:
                m["players"], m["blocks"] = m["blocks"], []
            return
    if armed:
        for p in m["players"]:
            if p[0] <= x < p[0] + 4 and p[1] <= y < p[1] + 4:
                for mk in m["markers"]:
                    mk[2] = False
                m["blocks"], m["players"] = m["players"], []
                return


def move_blocks(m, a):
    dx, dy = MOVES[a]
    blocks = sorted(m["blocks"])
    occupied = {cell_of(mk[0], mk[1], 2) for mk in m["markers"]}
    for i, b in enumerate(blocks):
        sx = dx if i == 0 else -dx
        nx, ny = b[0] + sx, b[1] + dy
        other = blocks[1 - i] if len(blocks) == 2 else None
        occ = set(occupied)
        if other is not None:
            occ.add((other[0], other[1]))
        if not cell_blocked(nx, ny, occ):
            b[0], b[1] = nx, ny
    m["blocks"] = blocks


def move_marker(m, a):
    dx, dy = MOVES[a]
    for mk in m["markers"]:
        if not mk[2]:
            continue
        occ = {cell_of(o[0], o[1], 2) for o in m["markers"] if o is not mk}
        occ |= {(p[0], p[1]) for p in m["players"]}
        cx, cy = cell_of(mk[0] + dx, mk[1] + dy, 2)
        if not cell_blocked(cx, cy, occ):
            mk[0] += dx
            mk[1] += dy


def render(m):
    items = []  # (sortkey, obj-without-name, prefix)
    for o in m["static"]:
        d = {k: v for k, v in o.items() if k != "name"}
        color = next(t for t in o["tags"] if t.startswith("color_"))[6:]
        items.append((d, "%s_%s" % (o["type"], color)))
    w = m["bar"]
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            items.append(({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                           "visible": True, "w": w, "x": bx, "y": by}, "wall_0"))
    for x, y, act in m["markers"]:
        c = 11 if act else 9
        items.append(({"h": 2, "layer": 1, "tags": ["color_%d" % c, "active" if act else "inactive", "marker"],
                       "type": "marker", "visible": True, "w": 2, "x": x, "y": y}, "marker_%d" % c))
    for x, y in m["players"]:
        items.append(({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                       "visible": True, "w": 4, "x": x, "y": y}, "player_1"))
    out = []
    base = 0
    if m["blocks"]:
        for nm, (x, y) in zip(("block_left", "block_right"), sorted(m["blocks"])):
            out.append({"h": 4, "layer": 1, "name": nm, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": x, "y": y})
        base = len(m["blocks"])
    items.sort(key=lambda it: (it[0]["y"], it[0]["x"]))
    for i, (d, prefix) in enumerate(items):
        d = dict(d)
        d["name"] = "%s_%d" % (prefix, base + i)
        out.append(d)
    return out


def transition_function(state, action):
    m = parse(state)
    last = _hidden["last"]
    n = _hidden["n"]
    if last is None or n is None or bar_width(n) != m["bar"] or last != m["bar"]:
        n = smallest_n(m["bar"])
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
    elif action in MOVES:
        if m["blocks"]:
            move_blocks(m, action)
        else:
            move_marker(m, action)
    n += 1
    m["bar"] = bar_width(n)
    _hidden["last"], _hidden["n"] = m["bar"], n
    return render(m)
