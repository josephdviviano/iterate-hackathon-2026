# Mechanics: blocks mode = two cyan 4x4 blocks (block_left/right); ACTION1/2 move both y-4/+4, ACTION3/4 move them
# apart/together (mirrored x). A move is blocked by bounds, markers, the other block or an invisible maze cell.
# Clicking an inactive marker -> armed mode (blocks become color_1 players, that marker active); ACTION1-4 move the
# active marker; clicking a player -> blocks mode. Bars wall_0 grow: width floor(3(n+1)/7), n = actions since level start.
# Unconfirmed: invisible maze cells inferred from blocked moves only; ACTION5/7 assumed no-ops; names = rank by (y,x).
import copy

STEP = 4
BOARD = (0, 1, 64, 63)  # x0, y0, x1, y1 playfield bounds
MAZE_CELLS = {(14, 38), (6, 18), (38, 14)}  # inferred invisible 4x4 maze cells (top-left corner of a cell)
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
_hidden = {"bar": None, "n": 0}


def bar_width(n):
    return 3 * (n + 1) // 7


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cell_of(r):
    return (r[0] - (r[0] - 2) % 4, r[1] - (r[1] - 2) % 4)


def free(rect, obstacles):
    x, y, w, h = rect
    if x < BOARD[0] or y < BOARD[1] or x + w > BOARD[2] or y + h > BOARD[3]:
        return False
    if cell_of(rect) in MAZE_CELLS:
        return False
    return not any(overlap(rect, o) for o in obstacles)


def parse(state):
    g = {"static": [], "markers": [], "blocks": {}, "players": [], "bar": 0}
    for o in state:
        t = o.get("type")
        if t == "player" and "block" in o.get("tags", []):
            g["blocks"][o["name"]] = [o["x"], o["y"]]
        elif t == "player":
            g["players"].append([o["x"], o["y"]])
        elif t == "marker":
            g["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o.get("tags", [])})
        elif t == "wall" and "color_0" in o.get("tags", []):
            if o["y"] == 0:
                g["bar"] = o["w"]
        else:
            g["static"].append(o)
    g["armed"] = bool(g["players"])
    if g["armed"]:
        ps = sorted(g["players"])
        g["blocks"] = {"block_left": ps[0], "block_right": ps[-1]}
    return g


def hit(o, x, y, w, h):
    return o[0] <= x < o[0] + w and o[1] <= y < o[1] + h


def step_blocks(g, a):
    dx, dy = DIRS[a]
    L, R = g["blocks"]["block_left"], g["blocks"]["block_right"]
    moves = {"block_left": (dx, dy), "block_right": (-dx, dy)}
    marks = [(m["x"], m["y"], 2, 2) for m in g["markers"]]
    new = {}
    for name, (mx, my) in moves.items():
        p = g["blocks"][name]
        other = R if name == "block_left" else L
        rect = (p[0] + mx, p[1] + my, 4, 4)
        new[name] = [rect[0], rect[1]] if free(rect, marks + [(other[0], other[1], 4, 4)]) else p
    g["blocks"] = new


def step_marker(g, a):
    dx, dy = DIRS[a]
    act = [m for m in g["markers"] if m["active"]]
    if not act:
        return
    m = act[0]
    obs = [(o["x"], o["y"], 2, 2) for o in g["markers"] if o is not m]
    obs += [(p[0], p[1], 4, 4) for p in g["blocks"].values()]
    rect = (m["x"] + dx, m["y"] + dy, 2, 2)
    if free(rect, obs):
        m["x"], m["y"] = rect[0], rect[1]


def click(g, x, y):
    for m in g["markers"]:
        if not m["active"] and hit((m["x"], m["y"]), x, y, 2, 2):
            for o in g["markers"]:
                o["active"] = o is m
            g["armed"] = True
            return
    if g["armed"] and any(hit(p, x, y, 4, 4) for p in g["blocks"].values()):
        g["armed"] = False
        for o in g["markers"]:
            o["active"] = False


def render(g, width):
    objs = [copy.deepcopy(o) for o in g["static"]]
    if width > 0:
        for bx, by in ((64 - width, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": width, "x": bx, "y": by})
    for m in g["markers"]:
        c, s = ("11", "active") if m["active"] and g["armed"] else ("9", "inactive")
        objs.append({"h": 2, "layer": 1, "tags": ["color_" + c, s, "marker"], "type": "marker",
                     "visible": True, "w": 2, "x": m["x"], "y": m["y"], "_c": c})
    fixed = []
    for name, (bx, by) in g["blocks"].items():
        if g["armed"]:
            objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                         "visible": True, "w": 4, "x": bx, "y": by, "_c": "1"})
        else:
            fixed.append({"h": 4, "layer": 1, "name": name, "pixels": [[10] * 4 for _ in range(4)],
                          "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                          "w": 4, "x": bx, "y": by})
    off = len(fixed)
    for i, o in enumerate(sorted(objs, key=lambda o: (o["y"], o["x"]))):
        color = o.pop("_c", None) or o["tags"][0].split("_", 1)[1]
        o["name"] = "%s_%s_%d" % (o["type"], color, i + off)
    return objs + fixed


def transition_function(state, action):
    g = parse(state)
    n = _hidden["n"] if _hidden["bar"] == g["bar"] else min(k for k in range(1000) if bar_width(k) >= g["bar"])
    a = action["action_id"] if isinstance(action, dict) else action
    if a == 6:
        click(g, action["x"], action["y"])
    elif a in DIRS:
        if g["armed"]:
            step_marker(g, a)
        else:
            step_blocks(g, a)
    n += 1
    w = bar_width(n)
    _hidden["bar"], _hidden["n"] = w, n
    return render(g, w)
