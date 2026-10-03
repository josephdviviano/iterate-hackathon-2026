# Mechanics: blocks (block_left/right) move together on a 4-grid: A1 y-4, A2 y+4, A3 apart, A4 together; each moves alone if its cell is free
#   (bounds [2,58], invisible maze cells, marker cells = marker xy-1, other block). Click marker -> blocks become armed players, marker active;
#   armed A1-A4 steer the active marker (blocked by maze/markers/players/bounds); click other marker = switch, click player = disarm, else no-op.
# Hidden counter a = actions since level start (continuity-gated; fallback a=(7w+2)//3 if bar else 0); bars w=3(a+1)//7 at top-right and bottom-left.
# Names: type_color_rank by (y,x); block mode reserves block_* and ranks others from 2, armed ranks all from 0. Hypothesis: maze cell (22,18) is invisible (step 124).
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
_mem = {"out": None, "a": 0, "left": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _prefix(name):
    return name.rsplit("_", 1)[0]


def parse(state):
    m = {"blocks": None, "players": None, "markers": [], "scenery": [], "bar": 0}
    blocks, players = {}, []
    for o in state:
        tags = o.get("tags", [])
        if o["name"] in ("block_left", "block_right"):
            blocks[o["name"]] = (o["x"], o["y"])
        elif o["type"] == "player":
            players.append((o["x"], o["y"]))
        elif o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in tags])
        elif o["type"] == "wall" and "color_0" in tags and o["h"] == 1:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["scenery"].append(o)
    if players:
        m["players"] = players
    else:
        m["blocks"] = [blocks.get("block_left"), blocks.get("block_right")]
    return m


def cell_free(c, blocked):
    return LO <= c[0] <= HI and LO <= c[1] <= HI and c not in MAZE and c not in blocked


def marker_cells(markers, skip=None):
    return {(mk[0] - 1, mk[1] - 1) for i, mk in enumerate(markers) if i != skip}


def move_blocks(m, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    l, r = m["blocks"]
    mc = marker_cells(m["markers"])
    nl, nr = (l[0] + dx, l[1] + dy), (r[0] - dx, r[1] + dy)
    nl = nl if cell_free(nl, mc | {r}) else l
    nr = nr if cell_free(nr, mc | {l}) and nr != nl else r
    m["blocks"] = [nl, nr]


def move_marker(m, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    for i, mk in enumerate(m["markers"]):
        if mk[2]:
            c = (mk[0] - 1 + dx, mk[1] - 1 + dy)
            blocked = marker_cells(m["markers"], skip=i) | set(m["players"])
            if cell_free(c, blocked):
                mk[0] += dx
                mk[1] += dy


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def click(m, x, y):
    for i, mk in enumerate(m["markers"]):
        if hit(x, y, mk[0], mk[1], 2):
            if mk[2]:
                return
            for mk2 in m["markers"]:
                mk2[2] = False
            mk[2] = True
            if m["blocks"] is not None:
                m["players"] = list(m["blocks"])
                m["blocks"] = None
            return
    if m["players"] is not None:
        for p in m["players"]:
            if hit(x, y, p[0], p[1], 4):
                m["blocks"] = list(m["players"])
                m["players"] = None
                for mk in m["markers"]:
                    mk[2] = False
                return


def render(m, a):
    out, ranked = [], []
    w = 3 * (a + 1) // 7
    for o in m["scenery"]:
        o = dict(o)
        ranked.append((o["y"], o["x"], _prefix(o["name"]), o))
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            o = {"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                 "visible": True, "w": w, "x": bx, "y": by}
            ranked.append((by, bx, "wall_0", o))
    for x, y, act in m["markers"]:
        col = "color_11" if act else "color_9"
        o = {"h": 2, "layer": 1, "tags": [col, "active" if act else "inactive", "marker"],
             "type": "marker", "visible": True, "w": 2, "x": x, "y": y}
        ranked.append((y, x, "marker_" + col[6:], o))
    if m["players"] is not None:
        for x, y in m["players"]:
            o = {"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                 "visible": True, "w": 4, "x": x, "y": y}
            ranked.append((y, x, "player_1", o))
        start = 0
    else:
        for nm, (x, y) in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"h": 4, "layer": 1, "name": nm, "pixels": [[10] * 4 for _ in range(4)],
                        "tags": ["cyan", "block", "player"], "type": "player",
                        "visible": True, "w": 4, "x": x, "y": y})
        start = 2
    ranked.sort(key=lambda t: (t[0], t[1]))
    for i, (_, _, pre, o) in enumerate(ranked):
        o["name"] = "%s_%d" % (pre, start + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["out"] is not None and _canon(state) == _mem["out"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] else 0
        left = None
    if m["players"] is not None:
        ps = sorted(m["players"])
        if left in ps:
            ps.remove(left)
            ps = [left] + ps
        m["players"] = ps
    if isinstance(action, dict):
        click(m, action.get("x"), action.get("y"))
    elif action in (1, 2, 3, 4):
        if m["blocks"] is not None:
            move_blocks(m, action)
        else:
            move_marker(m, action)
    a += 1
    out = render(m, a)
    _mem["out"] = _canon(out)
    _mem["a"] = a
    _mem["left"] = m["players"][0] if m["players"] is not None else None
    return out
