# Player steps by its size; destination cells of colour 4 or 5 block movement.
# A full colour-1 strip beside the destination pushes away until the next wall.
# Actions spend two budget columns; fully covered refuel rings refill the bar.
# Covered diamonds rotate the HUD clockwise; their sprites persist underneath.
# Underlay memory is continuity-gated; recolouring and chamber unlocking are hypotheses.

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
    return (o["x"] >= x and o["y"] >= y and
            o["x"] + o["w"] <= x + w and o["y"] + o["h"] <= y + h)


def overlaps(o, x, y, w, h, pad=0):
    return not (o["x"] - pad >= x + w or x >= o["x"] + o["w"] + pad or
                o["y"] - pad >= y + h or y >= o["y"] + o["h"] + pad)


def blocked(under, x, y, w, h):
    if x < 0 or y < 0 or x + w > 64 or y + h > 64:
        return True
    return any(under[yy][xx] in WALLS
               for yy in range(y, y + h) for xx in range(x, x + w))


def strip_side(under, x, y, w, h):
    def full(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and under[cy][cx] == STRIP
                   for cx, cy in cells)
    if full([(x - 1, yy) for yy in range(y, y + h)]):
        return (1, 0)
    if full([(x + w, yy) for yy in range(y, y + h)]):
        return (-1, 0)
    if full([(xx, y - 1) for xx in range(x, x + w)]):
        return (0, 1)
    if full([(xx, y + h) for xx in range(x, x + w)]):
        return (0, -1)
    return None


def bar_track(grid):
    for y in range(63, -1, -1):
        row = grid[y]
        xs = [x for x in range(64) if row[x] == BAR]
        if len(xs) >= 2:
            lo, hi = xs[0], xs[-1]
            while lo - 1 >= 0 and row[lo - 1] in (BAR, FLOOR):
                lo -= 1
            while hi + 1 < 64 and row[hi + 1] in (BAR, FLOOR):
                hi += 1
            rows = [yy for yy in (y - 1, y)
                    if all(grid[yy][xx] in (BAR, FLOOR) for xx in range(lo, hi + 1))]
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


def glyph_blocks(grid, g):
    bw, bh = g["w"] // 3, g["h"] // 3
    return [[grid[g["y"] + i * bh][g["x"] + j * bw]
             for j in range(3)] for i in range(3)], bw, bh


def glyph_write(grid, g, blocks, bw, bh):
    for i in range(3):
        for j in range(3):
            for dy in range(bh):
                for dx in range(bw):
                    grid[g["y"] + i * bh + dy][g["x"] + j * bw + dx] = blocks[i][j]


def glyph_rotate(grid, g):
    b, bw, bh = glyph_blocks(grid, g)
    glyph_write(grid, g, [[b[2 - j][i] for j in range(3)] for i in range(3)], bw, bh)


def b_bg(b):
    vals = [v for r in b for v in r]
    return 5 if 5 in vals else max(set(vals), key=vals.count)


def glyph_recolor(grid, g, colour):
    b, bw, bh = glyph_blocks(grid, g)
    bg = b_bg(b)
    glyph_write(grid, g, [[colour if v != bg else v for v in r] for r in b], bw, bh)


def glyph_matches_legend(grid, g, legend):
    b, _, _ = glyph_blocks(grid, g)
    bg = b_bg(b)
    lw, lh = legend["w"] // 3 or 1, legend["h"] // 3 or 1
    lb = [[grid[legend["y"] + i * lh][legend["x"] + j * lw]
           for j in range(3)] for i in range(3)]
    lbg = b_bg(lb)
    return all((b[i][j] != bg) == (lb[i][j] != lbg) and
               (b[i][j] == bg or b[i][j] == lb[i][j])
               for i in range(3) for j in range(3))


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


def update_player(state, grid, under, p, aid):
    px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]
    if aid not in MOVES:
        return px, py, False
    dx, dy = MOVES[aid]
    tx, ty = px + dx * pw, py + dy * ph
    if blocked(under, tx, ty, pw, ph):
        return px, py, False
    if chamber_locked(state, grid, tx, ty, pw, ph):
        return px, py, True
    push = strip_side(under, tx, ty, pw, ph)
    if push:
        while not blocked(under, tx + push[0] * pw, ty + push[1] * ph, pw, ph):
            tx, ty = tx + push[0] * pw, ty + push[1] * ph
    return tx, ty, False


def update_refuel(grid, o, glyph, track):
    paint(grid, o, FLOOR)
    bar_refill(grid, track)


def update_target(grid, o, glyph, track):
    if has_tag(o, "collect") and glyph is not None:
        glyph_rotate(grid, glyph)


def update_button(grid, o, glyph, track):
    if has_tag(o, "recolor") and glyph is not None:
        paint(grid, o, FLOOR)
        glyph_recolor(grid, glyph, 9)


def update_counter(grid, track):
    bar_burn(grid, track)


PICKUP_RULES = {"refuel": update_refuel, "target": update_target, "button": update_button}


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
        paint(under, p, FLOOR)
    aid = action.get("action_id") if isinstance(action, dict) else action
    nx, ny, free_reject = update_player(state, grid, under, p, aid)
    if free_reject:
        return grid
    track = bar_track(under)
    update_counter(under, track)
    glyph = next(iter(find(state, lambda o: has_tag(o, "progress"))), None)
    if (nx, ny) != (px, py):
        for o in state:
            rule = PICKUP_RULES.get(o.get("type"))
            if rule is not None and inside(o, nx, ny, pw, ph):
                rule(under, o, glyph, track)
    out = copy_grid(under)
    for j in range(ph):
        for i in range(pw):
            out[ny + j][nx + i] = sprite[j][i]
    _mem["frame"] = copy_grid(out)
    _mem["under"] = under
    return out
