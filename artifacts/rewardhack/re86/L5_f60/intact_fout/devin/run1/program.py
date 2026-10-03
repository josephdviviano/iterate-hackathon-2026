# Mechanics: outline shapes (X = diagonals, plus = cross, diamond = |dx|+|dy|=r) of one colour on a
# static floor (5) with boxes and 6x6 recolor swatches. A1-4 move the active shape by 3 (up/down/left/right);
# A5 makes the next shape (list order) active. A shape with a pixel inside a swatch takes the swatch colour.
# Shapes draw in list order (later on top); active centre = 0 in its own layer. HUD row 63: bar of 1s from the
# right, length = actions//4. Unconfirmed: action-count phase on a fresh state (assumed 4*bar+2), bounds/blocking, A6/A7.
MOVES = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
KINDS = ("X", "plus", "diamond")
FLOOR, HUD_BG, HUD_FG, CENTRE = 5, 15, 1, 0
_mem = {}


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_") and t[6:].isdigit():
            return int(t[6:])
    return None


def offsets(kind, r):
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if kind == "X" and abs(dx) == abs(dy):
                out.append((dx, dy))
            elif kind == "plus" and (dx == 0 or dy == 0):
                out.append((dx, dy))
            elif kind == "diamond" and abs(dx) + abs(dy) == r:
                out.append((dx, dy))
    return out


def shape_cells(s):
    return [(s["cx"] + dx, s["cy"] + dy) for dx, dy in offsets(s["kind"], s["r"])]


def inside(x, y):
    return 0 <= x < 64 and 0 <= y < 63


def fit_kind(frame, cx, cy, col):
    best = ("X", 1, -1)
    for kind in KINDS:
        r = 0
        while r < 40:
            ring = [(cx + dx, cy + dy) for dx, dy in offsets(kind, r + 1)
                    if max(abs(dx), abs(dy)) == r + 1 or kind == "diamond"]
            vis = [(x, y) for x, y in ring if inside(x, y)]
            if not vis or sum(frame[y][x] != col for x, y in vis) > 1:
                break
            r += 1
        if r > best[2]:
            best = (kind, r, r)
    return best[0], best[1]


def parse(state, frame):
    shapes = []
    players = sorted((o for o in state if o.get("type") == "player"),
                     key=lambda o: int(o["name"].split("_")[1]) if o["name"].split("_")[1].isdigit() else 0)
    zero = [(x, y) for y in range(63) for x in range(64) if frame[y][x] == CENTRE]
    for o in players:
        tags = o.get("tags", [])
        col = color_of(o)
        cx, cy = o.get("cx", o["x"] + o["w"] // 2), o.get("cy", o["y"] + o["h"] // 2)
        if "active" in tags and zero:
            cx, cy = zero[0]
        kind = next((k for k in KINDS if k in tags), None)
        if kind and o["w"] == o["h"]:
            r = (o["w"] - 1) // 2
        else:
            kind, r = fit_kind(frame, cx, cy, col)
        shapes.append({"kind": kind, "r": r, "col": col, "cx": cx, "cy": cy, "active": "active" in tags})
    swatches = [(o["x"], o["y"], o["w"], o["h"], color_of(o)) for o in state if o.get("type") == "swatch"]
    covered = set()
    for s in shapes:
        covered.update(c for c in shape_cells(s) if inside(*c))
        if s["active"]:
            covered.add((s["cx"], s["cy"]))
    bg = [row[:] for row in frame]
    for y in range(63):
        for x in range(64):
            if (x, y) in covered:
                bg[y][x] = static_cell(state, x, y)
    bar = sum(1 for v in frame[63] if v == HUD_FG)
    active = next((i for i, s in enumerate(shapes) if s["active"]), 0)
    return {"shapes": shapes, "active": active, "swatches": swatches, "bg": bg, "n": 4 * bar + 2}


def static_cell(state, x, y):
    for o in state:
        if o.get("type") not in ("target", "swatch"):
            continue
        ox, oy, w, h = o["x"], o["y"], o["w"], o["h"]
        if ox <= x < ox + w and oy <= y < oy + h:
            border = x in (ox, ox + w - 1) or y in (oy, oy + h - 1)
            col = color_of(o)
            if o["type"] == "target":
                return 4 if border else (col if col is not None else FLOOR)
            return 2 if border else (col if col is not None else FLOOR)
    return FLOOR


def touches_swatch(s, sw):
    x0, y0, w, h, col = sw
    return any(x0 <= x < x0 + w and y0 <= y < y0 + h for x, y in shape_cells(s))


def step(m, action):
    m["n"] += 1
    shapes = m["shapes"]
    if action == 5 and shapes:
        m["active"] = (m["active"] + 1) % len(shapes)
    elif action in MOVES and shapes:
        s = shapes[m["active"]]
        dx, dy = MOVES[action]
        s["cx"] += dx
        s["cy"] += dy
        for sw in m["swatches"]:
            if sw[4] is not None and touches_swatch(s, sw):
                s["col"] = sw[4]


def render(m):
    out = [row[:] for row in m["bg"]]
    for i, s in enumerate(m["shapes"]):
        for x, y in shape_cells(s):
            if inside(x, y):
                out[y][x] = s["col"]
        if i == m["active"] and inside(s["cx"], s["cy"]):
            out[s["cy"]][s["cx"]] = CENTRE
    bar = min(64, m["n"] // 4)
    out[63] = [HUD_BG] * (64 - bar) + [HUD_FG] * bar
    return out


def copy_model(m):
    return {"shapes": [dict(s) for s in m["shapes"]], "active": m["active"],
            "swatches": list(m["swatches"]), "bg": [r[:] for r in m["bg"]], "n": m["n"]}


def transition_function(state, action, frame):
    if isinstance(action, dict):
        action = action.get("action_id")
    if _mem.get("frame") == frame:
        m = copy_model(_mem["model"])
    else:
        m = parse(state, frame)
    step(m, action)
    out = render(m)
    _mem["frame"] = [r[:] for r in out]
    _mem["model"] = copy_model(m)
    return out
