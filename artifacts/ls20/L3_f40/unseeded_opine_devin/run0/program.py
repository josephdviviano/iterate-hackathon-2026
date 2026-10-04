# Mechanics: 5x5 player steps 5 cells (A1 up, A2 down, A3 left, A4 right); a dest box with any wall 4 /
# void 5 cell or off-board is blocked. A full 5-cell colour-1 strip just outside a dest edge is a conveyor
# that slides the player away from it until blocked. Every action burns the two leftmost 11 bar cells to 3;
# a refuel fully covered is consumed and refills the bar; a covered 'collect' rotates the HUD glyph 90 CW.
# Hidden state: under-player underlay (continuity-gated). Unconfirmed: recolor token, chamber unlock rule.

WALLS = (4, 5)
FLOOR = 3
BAR = 11
STRIP = 1
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"frame": None, "under": None}


def copy_grid(g):
    return [list(r) for r in g]


def find(state, pred):
    return [o for o in state if pred(o)]


def has_tag(o, t):
    return t in (o.get("tags") or [])


def player_of(state):
    ps = find(state, lambda o: o.get("type") == "player")
    return ps[0] if ps else None


def inside(o, x, y, w, h):
    return (o["x"] >= x and o["y"] >= y and o["x"] + o["w"] <= x + w and o["y"] + o["h"] <= y + h)


def overlaps(o, x, y, w, h, pad=0):
    return not (o["x"] - pad >= x + w or x >= o["x"] + o["w"] + pad or
                o["y"] - pad >= y + h or y >= o["y"] + o["h"] + pad)


def blocked(under, x, y, w, h):
    if x < 0 or y < 0 or x + w > 64 or y + h > 64:
        return True
    return any(under[yy][xx] in WALLS for yy in range(y, y + h) for xx in range(x, x + w))


def strip_side(under, x, y, w, h):
    """Return the push direction of a conveyor strip on an edge of the box, else None."""
    def full(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and under[cy][cx] == STRIP for cx, cy in cells)
    if full([(x - 1, yy) for yy in range(y, y + h)]):
        return (1, 0)
    if full([(x + w, yy) for yy in range(y, y + h)]):
        return (-1, 0)
    if full([(xx, y - 1) for xx in range(x, x + w)]):
        return (0, 1)
    if full([(xx, y + h) for xx in range(x, x + w)]):
        return (0, -1)
    return None


# ---- HUD: step bar -------------------------------------------------------
def bar_track(grid):
    """Cells of the budget track on the bar rows: runs of 11/3 between 5 delimiters."""
    for y in range(63, -1, -1):
        row = grid[y]
        xs = [x for x in range(64) if row[x] == BAR]
        if len(xs) >= 2:
            lo, hi = xs[0], xs[-1]
            while lo - 1 >= 0 and row[lo - 1] in (BAR, FLOOR):
                lo -= 1
            while hi + 1 < 64 and row[hi + 1] in (BAR, FLOOR):
                hi += 1
            rows = [yy for yy in (y - 1, y) if all(grid[yy][xx] in (BAR, FLOOR) for xx in range(lo, hi + 1))]
            return rows, lo, hi
    return None


def bar_burn(grid, track, n=2):
    if not track:
        return
    rows, lo, hi = track
    xs = [x for x in range(lo, hi + 1) if grid[rows[0]][x] == BAR][:n]
    for yy in rows:
        for x in xs:
            grid[yy][x] = FLOOR


def bar_refill(grid, track):
    if not track:
        return
    rows, lo, hi = track
    for yy in rows:
        for x in range(lo, hi + 1):
            grid[yy][x] = BAR


# ---- HUD: glyph ----------------------------------------------------------
def glyph_blocks(grid, g):
    bw, bh = g["w"] // 3, g["h"] // 3
    return [[grid[g["y"] + i * bh][g["x"] + j * bw] for j in range(3)] for i in range(3)], bw, bh


def glyph_write(grid, g, blocks, bw, bh):
    for i in range(3):
        for j in range(3):
            for dy in range(bh):
                for dx in range(bw):
                    grid[g["y"] + i * bh + dy][g["x"] + j * bw + dx] = blocks[i][j]


def glyph_rotate(grid, g):
    b, bw, bh = glyph_blocks(grid, g)
    glyph_write(grid, g, [[b[2 - j][i] for j in range(3)] for i in range(3)], bw, bh)


def glyph_recolor(grid, g, colour):
    b, bw, bh = glyph_blocks(grid, g)
    bg = b_bg(b)
    glyph_write(grid, g, [[colour if v != bg else v for v in r] for r in b], bw, bh)


def b_bg(b):
    vals = [v for r in b for v in r]
    return 5 if 5 in vals else max(set(vals), key=vals.count)


def glyph_matches_legend(grid, g, legend):
    b, _, _ = glyph_blocks(grid, g)
    bg = b_bg(b)
    lw, lh = legend["w"] // 3 or 1, legend["h"] // 3 or 1
    lb = [[grid[legend["y"] + i * lh][legend["x"] + j * lw] for j in range(3)] for i in range(3)]
    lbg = b_bg(lb)
    return all((b[i][j] != bg) == (lb[i][j] != lbg) and (b[i][j] == bg or b[i][j] == lb[i][j])
               for i in range(3) for j in range(3))


# ---- guards ----------------------------------------------------------------
def chamber_locked(state, grid, x, y, w, h):
    glyph = next(iter(find(state, lambda o: has_tag(o, "progress"))), None)
    for c in find(state, lambda o: has_tag(o, "chamber")):
        if overlaps(c, x, y, w, h, pad=3):
            if glyph is None or not glyph_matches_legend(grid, glyph, c):
                return True
    return False


def paint(grid, o, colour):
    for yy in range(o["y"], o["y"] + o["h"]):
        for xx in range(o["x"], o["x"] + o["w"]):
            grid[yy][xx] = colour


# ---- main ------------------------------------------------------------------
def transition_function(state, action, frame):
    grid = copy_grid(frame)
    p = player_of(state)
    if p is None:
        return grid
    px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]
    sprite = [[frame[py + j][px + i] for i in range(pw)] for j in range(ph)]

    if _mem["frame"] == frame and _mem["under"] is not None:
        under = copy_grid(_mem["under"])
    else:
        under = copy_grid(frame)
        for yy in range(py, py + ph):
            for xx in range(px, px + pw):
                under[yy][xx] = FLOOR

    aid = action.get("action_id") if isinstance(action, dict) else action
    nx, ny = px, py
    if aid in MOVES:
        dx, dy = MOVES[aid]
        tx, ty = px + dx * pw, py + dy * ph
        if not blocked(under, tx, ty, pw, ph) and chamber_locked(state, grid, tx, ty, pw, ph):
            return grid
        if not blocked(under, tx, ty, pw, ph):
            nx, ny = tx, ty
            push = strip_side(under, nx, ny, pw, ph)
            if push:
                while not blocked(under, nx + push[0] * pw, ny + push[1] * ph, pw, ph):
                    nx, ny = nx + push[0] * pw, ny + push[1] * ph

    track = bar_track(under)
    bar_burn(under, track)
    glyph = next(iter(find(state, lambda o: has_tag(o, "progress"))), None)
    if (nx, ny) != (px, py):
        for o in state:
            if o is p or not inside(o, nx, ny, pw, ph):
                continue
            if o.get("type") == "refuel":
                paint(under, o, FLOOR)
                bar_refill(under, track)
            elif has_tag(o, "collect") and glyph is not None:
                glyph_rotate(under, glyph)
            elif has_tag(o, "recolor") and glyph is not None:
                paint(under, o, FLOOR)
                glyph_recolor(under, glyph, 9)

    out = copy_grid(under)
    for j in range(ph):
        for i in range(pw):
            out[ny + j][nx + i] = sprite[j][i]
    _mem["frame"] = copy_grid(out)
    _mem["under"] = under
    return out

