# Mechanics: two cyan blocks (block mode) move together: A1/A2 y-4/y+4, A3 apart, A4 together (mirrored x);
# each block is blocked individually by its half's bounds, inferred maze cells and marker cells.
# Clicking an inactive marker activates it (color_11) and arms the blocks (color_1 players); in armed mode
# A1-A4 move the active marker by 4 (blocked by maze/marker/player cells); clicking a player disarms.
# Bars (color_0) have w=floor(3(a+1)/7), a = actions since level start (continuity-gated; hypothesis: maze beyond inferred cells unknown).
import json

MAZE = {(6, 18), (14, 38), (38, 14)}
_last = {"canon": None, "a": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _color(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def _inside(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def parse(state):
    m = {"blocks": [], "players": [], "markers": [], "static": [], "bar": 0}
    for o in state:
        tags = o.get("tags", [])
        if "block" in tags:
            m["blocks"].append([o["x"], o["y"]])
        elif "armed" in tags:
            m["players"].append([o["x"], o["y"]])
        elif o["type"] == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in tags})
        elif o["type"] == "wall" and _color(o) == 0:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["static"].append(dict(o))
    return m


def _bar_width(a):
    return (3 * (a + 1)) // 7


def _blocked_cells(m, exclude=None):
    cells = set(MAZE)
    for mk in m["markers"]:
        if mk is not exclude:
            cells.add((mk["x"] - 1, mk["y"] - 1))
    for p in m["players"]:
        cells.add((p[0], p[1]))
    return cells


def _move_blocks(m, action):
    cells = _blocked_cells(m)
    for b in m["blocks"]:
        left = b[0] < 32
        dx, dy = 0, 0
        if action == 1:
            dy = -4
        elif action == 2:
            dy = 4
        elif action == 3:
            dx = -4 if left else 4
        elif action == 4:
            dx = 4 if left else -4
        nx, ny = b[0] + dx, b[1] + dy
        lo, hi = (2, 26) if left else (34, 58)
        if lo <= nx <= hi and 2 <= ny <= 58 and (nx, ny) not in cells:
            b[0], b[1] = nx, ny


def _move_marker(m, action):
    act = [mk for mk in m["markers"] if mk["active"]]
    if not act:
        return
    mk = act[0]
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    cx, cy = mk["x"] - 1 + dx, mk["y"] - 1 + dy
    if 2 <= cx <= 58 and 2 <= cy <= 58 and (cx, cy) not in _blocked_cells(m, exclude=mk):
        mk["x"], mk["y"] = cx + 1, cy + 1


def _click(m, x, y):
    for mk in m["markers"]:
        if mk["x"] <= x < mk["x"] + 2 and mk["y"] <= y < mk["y"] + 2:
            if mk["active"]:
                return
            for o in m["markers"]:
                o["active"] = o is mk
            if m["blocks"]:
                m["players"], m["blocks"] = m["blocks"], []
            return
    for p in m["players"]:
        if p[0] <= x < p[0] + 4 and p[1] <= y < p[1] + 4:
            m["blocks"], m["players"] = m["players"], []
            for o in m["markers"]:
                o["active"] = False
            return


def step(m, action):
    if isinstance(action, dict):
        _click(m, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if m["blocks"]:
            _move_blocks(m, action)
        else:
            _move_marker(m, action)


def render(m, a):
    objs = []  # (obj, fixed_name)
    for b in m["blocks"]:
        nm = "block_left" if b[0] < 32 else "block_right"
        objs.append(({"h": 4, "w": 4, "layer": 1, "x": b[0], "y": b[1], "type": "player", "visible": True,
                      "tags": ["cyan", "block", "player"], "pixels": [[10] * 4 for _ in range(4)]}, nm))
    for p in m["players"]:
        objs.append(({"h": 4, "w": 4, "layer": 1, "x": p[0], "y": p[1], "type": "player", "visible": True,
                      "tags": ["color_1", "armed", "player"]}, None))
    for mk in m["markers"]:
        c = 11 if mk["active"] else 9
        objs.append(({"h": 2, "w": 2, "layer": 1, "x": mk["x"], "y": mk["y"], "type": "marker", "visible": True,
                      "tags": ["color_%d" % c, "active" if mk["active"] else "inactive", "marker"]}, None))
    for s in m["static"]:
        objs.append((s, None))
    w = _bar_width(a)
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append(({"h": 1, "w": w, "layer": 0, "x": bx, "y": by, "type": "wall", "visible": True,
                          "tags": ["color_0", "wall"]}, None))
    out = []
    rank = sum(1 for _, nm in objs if nm is not None)
    for o, nm in sorted(objs, key=lambda t: (t[0]["y"], t[0]["x"])):
        o = dict(o)
        if nm is None:
            o["name"] = "%s_%d_%d" % (o["type"], _color(o), rank)
            rank += 1
        else:
            o["name"] = nm
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _last["canon"] is not None and _canon(state) == _last["canon"]:
        a = _last["a"]
    else:
        a = (7 * m["bar"] - 1) // 3 if m["bar"] > 0 else 0
    step(m, action)
    a += 1
    out = render(m, a)
    _last["canon"], _last["a"] = _canon(out), a
    return out
