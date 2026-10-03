# Mechanics: 5x5 player steps 5 cells (A1-4); dest box with any wall/void cell (4/5) or off-board is a blocked no-op.
# Every action burns 2 cells from the left of the colour-11 step bar (rows 61-62); covering a refuel ring consumes it and refills the bar.
# A full colour-1 strip on an edge of the arrival cell is a conveyor: the player slides away from it until blocked.
# Covering a 'collect' target rotates the 3x3 HUD glyph 90deg CW; the target is hidden under the player and restored on leaving (continuity memory).
# Hypotheses (unobserved here): recolor token -> glyph colour 9; dest touching legend chamber while glyph != legend -> free refusal; clicks burn 2.
WALL, VOID, FLOOR, BAR, STRIP = 4, 5, 3, 11, 1
DIRS = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
_mem = {"frame": None, "under": []}


def find(state, pred):
    return [o for o in state if pred(o)]


def blocked(frame, x, y, w, h):
    if x < 0 or y < 0 or x + w > 64 or y + h > 64:
        return True
    return any(frame[yy][xx] in (WALL, VOID) for yy in range(y, y + h) for xx in range(x, x + w))


def conveyor_dir(frame, x, y, w, h):
    def strip(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and frame[cy][cx] == STRIP for cx, cy in cells)
    if strip([(x - 1, yy) for yy in range(y, y + h)]):
        return (w, 0)
    if strip([(x + w, yy) for yy in range(y, y + h)]):
        return (-w, 0)
    if strip([(xx, y - 1) for xx in range(x, x + w)]):
        return (0, h)
    if strip([(xx, y + h) for xx in range(x, x + w)]):
        return (0, -h)
    return None


def covers(px, py, pw, ph, o):
    return px <= o["x"] and py <= o["y"] and o["x"] + o["w"] <= px + pw and o["y"] + o["h"] <= py + ph


def overlaps(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def burn(out, refill=False):
    row = out[61]
    cells = [x for x in range(64) if row[x] == BAR]
    if not cells:
        return
    right = max(cells)
    start = right
    while start - 1 >= 0 and row[start - 1] in (BAR, FLOOR):
        start -= 1
    left = min(cells)
    for yy in (61, 62):
        for xx in range(left, min(left + 2, right + 1)):
            out[yy][xx] = FLOOR
        if refill:
            for xx in range(start, right + 1):
                out[yy][xx] = BAR


def glyph_grid(frame, g):
    s = g["w"] // 3
    grid = [[frame[g["y"] + r * s][g["x"] + c * s] for c in range(3)] for r in range(3)]
    return grid, s


def paint_glyph(out, g, grid, s):
    for r in range(3):
        for c in range(3):
            for dy in range(s):
                for dx in range(s):
                    out[g["y"] + r * s + dy][g["x"] + c * s + dx] = grid[r][c]


def rotate_glyph(out, g):
    grid, s = glyph_grid(out, g)
    rot = [[grid[2 - c][r] for c in range(3)] for r in range(3)]
    paint_glyph(out, g, rot, s)


def recolor_glyph(out, g, colour):
    grid, s = glyph_grid(out, g)
    paint_glyph(out, g, [[colour if v != VOID else v for v in row] for row in grid], s)


def glyph_matches_legend(frame, g, legend):
    grid, _ = glyph_grid(frame, g)
    mask = [t for t in legend.get("tags", []) if set(t) <= {"#", "."} and len(t) == 9]
    if not mask:
        return False
    lx, ly = legend["x"], legend["y"]
    lc = [frame[ly + i // 3][lx + i % 3] for i in range(9)]
    fill = [v for v, m in zip(lc, mask[0]) if m == "#"]
    colour = fill[0] if fill else None
    for i, m in enumerate(mask[0]):
        v = grid[i // 3][i % 3]
        if (m == "#") != (v != VOID) or (m == "#" and v != colour):
            return False
    return True


def transition_function(state, action, frame):
    out = [list(map(int, row)) for row in frame]
    under = _mem["under"] if _mem["frame"] == frame else []
    _mem["under"] = under
    players = find(state, lambda o: o.get("type") == "player")
    glyphs = find(state, lambda o: o.get("type") == "counter" and "progress" in o.get("tags", []))
    legends = find(state, lambda o: "chamber" in o.get("tags", []))
    if not players or action not in DIRS:
        burn(out)
        _mem["frame"] = out
        return out
    p = players[0]
    px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]
    dx, dy = DIRS[action]
    nx, ny = px + dx, py + dy
    for lg in legends:
        if overlaps(nx, ny, pw, ph, lg["x"] - 3, lg["y"] - 3, lg["w"] + 6, lg["h"] + 6) and not (
                glyphs and glyph_matches_legend(frame, glyphs[0], lg)):
            _mem["frame"] = out
            return out
    if blocked(frame, nx, ny, pw, ph):
        burn(out)
        _mem["frame"] = out
        return out
    push = conveyor_dir(frame, nx, ny, pw, ph)
    if push:
        while not blocked(frame, nx + push[0], ny + push[1], pw, ph):
            nx, ny = nx + push[0], ny + push[1]
    sprite = [row[px:px + pw] for row in frame[py:py + ph]]
    refill = False
    new_under = []
    for o in state:
        if o is p or o.get("type") == "counter" or not covers(nx, ny, pw, ph, o):
            continue
        tags = o.get("tags", [])
        if o.get("type") == "refuel":
            refill = True
        elif "recolor" in tags:
            if glyphs:
                recolor_glyph(out, glyphs[0], 9)
        elif "collect" in tags:
            if glyphs:
                rotate_glyph(out, glyphs[0])
            new_under.append([(xx, yy, frame[yy][xx]) for yy in range(o["y"], o["y"] + o["h"])
                              for xx in range(o["x"], o["x"] + o["w"])])
    for yy in range(py, py + ph):
        for xx in range(px, px + pw):
            out[yy][xx] = FLOOR
    for cells in under:
        for xx, yy, v in cells:
            out[yy][xx] = v
    for yy in range(ph):
        for xx in range(pw):
            out[ny + yy][nx + xx] = sprite[yy][xx]
    burn(out, refill)
    _mem["under"] = new_under
    _mem["frame"] = out
    return out
