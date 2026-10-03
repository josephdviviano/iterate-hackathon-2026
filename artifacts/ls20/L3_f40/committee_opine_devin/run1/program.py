# Mechanics: 5x5 player steps 5 cells (A1-4); dest box containing wall colour 4 or void 5 is blocked.
# Colour-1 conveyor strip on a dest edge pushes the player away from it until blocked. Every action
# burns 2 cells of the colour-11 budget bar (rows 61-62); fully covering a refuel ring consumes it and refills.
# Covering a 'collect' target rotates the 3x3 HUD glyph 90 deg CW (target is occluded, reappears when uncovered).
# Unconfirmed: recolor token (glyph 12->9) and legend-chamber free refusal while glyph != legend (unobserved).
WALL, VOID, FLOOR, BAR, CONV = 4, 5, 3, 11, 1
STEP = 5
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_memo = {"out": None, "under": None}


def find(state, typ=None, tag=None):
    return [o for o in state if (typ is None or o["type"] == typ) and (tag is None or tag in o.get("tags", []))]


def box(o):
    return o["x"], o["y"], o["w"], o["h"]


def covers(px, py, s, o):
    x, y, w, h = box(o)
    return px <= x and py <= y and x + w <= px + s and y + h <= py + s


def overlaps(px, py, s, x, y, w, h):
    return px < x + w and x < px + s and py < y + h and y < py + s


def blocked(under, px, py, s):
    if px < 0 or py < 0 or px + s > 64 or py + s > 64:
        return True
    return any(under[y][x] in (WALL, VOID) for y in range(py, py + s) for x in range(px, px + s))


def conveyor_push(under, px, py, s):
    """Direction a conveyor strip on an edge of the box pushes towards, or None."""
    def strip(cells):
        return all(0 <= x < 64 and 0 <= y < 64 and under[y][x] == CONV for x, y in cells)
    if strip([(px - 1, py + i) for i in range(s)]):
        return (1, 0)
    if strip([(px + s, py + i) for i in range(s)]):
        return (-1, 0)
    if strip([(px + i, py - 1) for i in range(s)]):
        return (0, 1)
    if strip([(px + i, py + s) for i in range(s)]):
        return (0, -1)
    return None


def bar_track(frame):
    row = frame[61]
    xs = [x for x in range(64) if row[x] == BAR]
    if not xs:
        xs = [x for x in range(64) if row[x] == FLOOR and row[x - 1] in (FLOOR, VOID)]
        if not xs:
            return []
    lo = hi = xs[0]
    while lo - 1 >= 0 and row[lo - 1] in (BAR, FLOOR):
        lo -= 1
    hi = xs[-1]
    while hi + 1 < 64 and row[hi + 1] in (BAR, FLOOR):
        hi += 1
    return list(range(lo, hi + 1))


def burn(out, track):
    full = [x for x in track if out[61][x] == BAR]
    for x in full[:2]:
        for y in (61, 62):
            out[y][x] = FLOOR


def refill(out, track):
    for x in track:
        for y in (61, 62):
            out[y][x] = BAR


def glyph_mask(frame, g):
    x0, y0 = g["x"], g["y"]
    m = [[frame[y0 + 2 * r][x0 + 2 * c] != VOID for c in range(3)] for r in range(3)]
    col = next((frame[y0 + 2 * r][x0 + 2 * c] for r in range(3) for c in range(3) if m[r][c]), 12)
    return m, col


def draw_glyph(out, g, m, col):
    x0, y0 = g["x"], g["y"]
    for r in range(3):
        for c in range(3):
            for dy in range(2):
                for dx in range(2):
                    out[y0 + 2 * r + dy][x0 + 2 * c + dx] = col if m[r][c] else VOID


def rotate_cw(m):
    n = len(m)
    return [[m[n - 1 - j][i] for j in range(n)] for i in range(n)]


def legend_mask(o):
    t = next((t for t in o.get("tags", []) if len(t) == 9 and set(t) <= {"#", "."}), None)
    return None if t is None else [[t[3 * r + c] == "#" for c in range(3)] for r in range(3)]


def chamber_rejects(state, frame, px, py, s):
    g = next(iter(find(state, "counter", "progress")), None)
    for o in find(state, tag="chamber"):
        x, y, w, h = box(o)
        if not overlaps(px, py, s, x - 3, y - 3, w + 6, h + 6):
            continue
        want = legend_mask(o)
        if g is None or want is None:
            return True
        m, col = glyph_mask(frame, g)
        if m != want or col != 9:
            return True
    return False


def underlay(frame, player):
    if _memo["out"] is not None and frame == _memo["out"]:
        return [r[:] for r in _memo["under"]]
    u = [r[:] for r in frame]
    if player:
        x, y, w, h = box(player)
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                u[yy][xx] = FLOOR
    return u


def transition_function(state, action, frame):
    player = next(iter(find(state, "player")), None)
    under = underlay(frame, player)
    out = [r[:] for r in frame]
    track = bar_track(frame)
    aid = action if isinstance(action, int) else action.get("action_id")
    if player is None or aid not in DIRS:
        burn(out, track)
        return finish(out, under)
    px, py, s = player["x"], player["y"], player["w"]
    dx, dy = DIRS[aid]
    nx, ny = px + dx * STEP, py + dy * STEP
    if blocked(under, nx, ny, s):
        burn(out, track)
        return finish(out, under)
    if chamber_rejects(state, frame, nx, ny, s):
        return finish(out, under)
    push = conveyor_push(under, nx, ny, s)
    if push:
        while not blocked(under, nx + push[0] * STEP, ny + push[1] * STEP, s):
            nx, ny = nx + push[0] * STEP, ny + push[1] * STEP
    # erase old player
    for y in range(py, py + s):
        for x in range(px, px + s):
            out[y][x] = under[y][x]
    burn(out, track)
    g = next(iter(find(state, "counter", "progress")), None)
    for o in find(state, "refuel"):
        if covers(nx, ny, s, o):
            x, y, w, h = box(o)
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    under[yy][xx] = FLOOR
            refill(out, track)
    for o in find(state, tag="collect"):
        if covers(nx, ny, s, o) and g is not None:
            m, col = glyph_mask(out, g)
            draw_glyph(out, g, rotate_cw(m), col)
    for o in find(state, tag="recolor"):
        if covers(nx, ny, s, o):
            x, y, w, h = box(o)
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    under[yy][xx] = FLOOR
            if g is not None:
                m, _ = glyph_mask(out, g)
                draw_glyph(out, g, m, 9)
    pix = player.get("pixels")
    for j in range(s):
        for i in range(s):
            out[ny + j][nx + i] = pix[j][i] if pix else 12
    return finish(out, under)


def finish(out, under):
    _memo["out"] = [r[:] for r in out]
    _memo["under"] = under
    return out
