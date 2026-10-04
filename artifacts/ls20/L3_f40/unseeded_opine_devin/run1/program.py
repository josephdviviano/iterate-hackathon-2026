# Mechanics: player 5x5 moves 5 cells per arrow; dest blocked if off-board or any wall (4) / void (5) cell.
# Conveyor: a 5-cell colour-1 strip just outside a dest edge slides the player away until blocked.
# step_bar burns its 2 leftmost bar cells (11->3) per arrow incl. blocked ones; a covered ring refills it.
# Covering a 'collect' object rotates the 3x3-of-2x2 HUD glyph 90 CW; the object is restored when uncovered
# (continuity-gated memory). Hypothesis (unobserved here): chamber reject while glyph != legend; recolor -> 9.
WALL, VOID, FLOOR, BAR, CONVEYOR = 4, 5, 3, 11, 1
DIRS = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
RECOLOR = {"hud_blue": 9}
_mem = {"out": None, "hidden": []}


def find(state, typ=None, tag=None):
    return [o for o in state if (typ is None or o.get("type") == typ) and (tag is None or tag in o.get("tags", []))]


def box_cells(x, y, w, h):
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def passable(frame, x, y, w, h):
    for i, j in box_cells(x, y, w, h):
        if not (0 <= i < 64 and 0 <= j < 64) or frame[j][i] in (WALL, VOID):
            return False
    return True


def conveyor_dir(frame, x, y, w, h):
    def strip(cells):
        return all(0 <= i < 64 and 0 <= j < 64 and frame[j][i] == CONVEYOR for i, j in cells)
    if strip([(x - 1, j) for j in range(y, y + h)]):
        return (w, 0)
    if strip([(x + w, j) for j in range(y, y + h)]):
        return (-w, 0)
    if strip([(i, y - 1) for i in range(x, x + w)]):
        return (0, h)
    if strip([(i, y + h) for i in range(x, x + w)]):
        return (0, -h)
    return None


def inside(o, x, y, w, h):
    return x <= o["x"] and y <= o["y"] and o["x"] + o["w"] <= x + w and o["y"] + o["h"] <= y + h


def overlaps(o, x, y, w, h):
    return o["x"] < x + w and x < o["x"] + o["w"] and o["y"] < y + h and y < o["y"] + o["h"]


def glyph_grid(frame, g):
    return [[frame[g["y"] + 2 * r][g["x"] + 2 * c] for c in range(3)] for r in range(3)]


def set_glyph(out, g, grid):
    for r in range(3):
        for c in range(3):
            for dy in range(2):
                for dx in range(2):
                    out[g["y"] + 2 * r + dy][g["x"] + 2 * c + dx] = grid[r][c]


def glyph_matches_legend(frame, g, legend):
    pattern = next((t for t in legend.get("tags", []) if len(t) == 9 and set(t) <= set("#.")), None)
    if g is None or pattern is None:
        return False
    grid = glyph_grid(frame, g)
    lcol = [frame[legend["y"] + j][legend["x"] + i] for i, j in box_cells(0, 0, 3, 3)
            if frame[legend["y"] + j][legend["x"] + i] not in (VOID, FLOOR)]
    for k, ch in enumerate(pattern):
        v = grid[k // 3][k % 3]
        if (ch == "#") != (v != VOID):
            return False
        if ch == "#" and lcol and v != lcol[0]:
            return False
    return True


def burn_bar(out, frame, bar, refill):
    y0, h = bar["y"], bar["h"]
    right = bar["x"] + bar["w"]
    left = bar["x"]
    while left - 1 >= 0 and frame[y0][left - 1] in (BAR, FLOOR):
        left -= 1
    filled = [i for i in range(left, right) if frame[y0][i] == BAR]
    for i in filled[:2]:
        for j in range(y0, y0 + h):
            out[j][i] = FLOOR
    if refill:
        for i in range(left, right):
            for j in range(y0, y0 + h):
                out[j][i] = BAR


def transition_function(state, action, frame):
    hidden = _mem["hidden"] if _mem["out"] == frame else []
    out = [row[:] for row in frame]
    players = find(state, "player")
    if not isinstance(action, int) or action not in DIRS or not players:
        _mem["out"], _mem["hidden"] = out, hidden
        return out
    p = players[0]
    px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]
    dx, dy = DIRS[action]
    nx, ny = px + dx, py + dy
    glyphs = find(state, "counter", "progress")
    glyph = glyphs[0] if glyphs else None
    moved = passable(frame, nx, ny, pw, ph)
    if moved:
        for legend in find(state, tag="chamber"):
            if overlaps(legend, nx, ny, pw, ph) and not glyph_matches_legend(frame, glyph, legend):
                _mem["out"], _mem["hidden"] = out, hidden
                return out
        slide = conveyor_dir(frame, nx, ny, pw, ph)
        if slide:
            while passable(frame, nx + slide[0], ny + slide[1], pw, ph):
                nx, ny = nx + slide[0], ny + slide[1]
    refill = False
    if moved:
        sprite = [[frame[py + j][px + i] for i in range(pw)] for j in range(ph)]
        for i, j in box_cells(px, py, pw, ph):
            out[j][i] = FLOOR
        keep = []
        for hx, hy, hw, hh, cells in hidden:
            if overlaps({"x": hx, "y": hy, "w": hw, "h": hh}, nx, ny, pw, ph):
                keep.append((hx, hy, hw, hh, cells))
            else:
                for (i, j), v in cells:
                    out[j][i] = v
        hidden = keep
        covered = [o for o in state if o is not p and inside(o, nx, ny, pw, ph)]
        for o in covered:
            tags = o.get("tags", [])
            if "budget" in tags and o.get("type") == "refuel":
                refill = True
            if "collect" in tags:
                hidden.append((o["x"], o["y"], o["w"], o["h"],
                               [((i, j), frame[j][i]) for i, j in box_cells(o["x"], o["y"], o["w"], o["h"])]))
                if glyph:
                    g = glyph_grid(out, glyph)
                    set_glyph(out, glyph, [[g[2 - c][r] for c in range(3)] for r in range(3)])
            if "recolor" in tags and glyph:
                col = next((RECOLOR[t] for t in tags if t in RECOLOR), None)
                if col is not None:
                    g = glyph_grid(out, glyph)
                    set_glyph(out, glyph, [[col if v != VOID else v for v in row] for row in g])
        for j in range(ph):
            for i in range(pw):
                out[ny + j][nx + i] = sprite[j][i]
    for bar in find(state, "counter", "budget"):
        burn_bar(out, frame, bar, refill)
    _mem["out"], _mem["hidden"] = out, hidden
    return out

