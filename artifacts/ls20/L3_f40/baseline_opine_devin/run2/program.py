# Mechanics: 5x5 player steps 5 cells (A1-4); a dest is blocked if out of bounds or any cell is wall colour 4.
# A 5-cell colour-1 strip on a dest edge is a conveyor: the player slides away from it until blocked.
# Every action burns 2 bar cells (colour 11 -> 3, from the left); a fully covered refuel ring is consumed and refills the bar.
# Covering a 'collect' target rotates the HUD glyph 90 CW (target stays, restored on uncover via continuity-gated memory);
# recolor button -> glyph colour 9; chamber (legend) dest rejected free while glyph != legend. Unconfirmed: unlock, token, click.
WALL, FLOOR, BAR, BELT = 4, 3, 11, 1
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"frame": None, "under": None}


def find(state, pred):
    return [o for o in state if pred(o)]


def has_tag(o, t):
    return t in (o.get("tags") or [])


def cell_ok(frame, x, y, s):
    if x < 0 or y < 0 or x + s > 64 or y + s > 64:
        return False
    return all(frame[y + j][x + i] != WALL for j in range(s) for i in range(s))


def belt_push(frame, x, y, s):
    edges = [((x - 1, y, 0, 1), (1, 0)), ((x + s, y, 0, 1), (-1, 0)),
             ((x, y - 1, 1, 0), (0, 1)), ((x, y + s, 1, 0), (0, -1))]
    for (ex, ey, ix, iy), d in edges:
        cells = [(ex + k * ix, ey + k * iy) for k in range(s)]
        if all(0 <= cx < 64 and 0 <= cy < 64 and frame[cy][cx] == BELT for cx, cy in cells):
            return d
    return None


def covered(o, x, y, s):
    return x <= o["x"] and y <= o["y"] and o["x"] + o["w"] <= x + s and o["y"] + o["h"] <= y + s


def overlaps(o, x, y, s):
    return o["x"] < x + s and x < o["x"] + o["w"] and o["y"] < y + s and y < o["y"] + o["h"]


def glyph_box(frame, g):
    return [(g["x"] + i, g["y"] + j) for j in range(g["h"]) for i in range(g["w"])]


def glyph_mask(frame, g):
    n = 3
    bw, bh = g["w"] // n, g["h"] // n
    mask, col = "", None
    for r in range(n):
        for c in range(n):
            v = frame[g["y"] + r * bh][g["x"] + c * bw]
            mask += "." if v == 5 else "#"
            if v != 5:
                col = v
    return mask, col


def rotate_glyph(out, frame, g):
    n = 3
    bw, bh = g["w"] // n, g["h"] // n
    for r in range(n):
        for c in range(n):
            sr, sc = n - 1 - c, r
            for j in range(bh):
                for i in range(bw):
                    out[g["y"] + r * bh + j][g["x"] + c * bw + i] = frame[g["y"] + sr * bh + j][g["x"] + sc * bw + i]


def recolor_glyph(out, g, colour):
    for x, y in glyph_box(out, g):
        if out[y][x] != 5:
            out[y][x] = colour


def legend_info(frame, lg):
    mask = next((t for t in lg.get("tags", []) if len(t) == 9 and set(t) <= set("#.")), None)
    cols = [frame[y][x] for x, y in glyph_box(frame, lg) if frame[y][x] != 5]
    return mask, (cols[0] if cols else None)


def burn(out, bar, n=2):
    y0 = bar["y"]
    xs = [x for x in range(64) if out[y0][x] == BAR]
    for x in sorted(xs)[:n]:
        for j in range(bar["h"]):
            out[y0 + j][x] = FLOOR


def refill(out, bar):
    y0 = bar["y"]
    right = bar["x"] + bar["w"]
    x = right - 1
    while x >= 0 and out[y0][x] in (BAR, FLOOR):
        for j in range(bar["h"]):
            out[y0 + j][x] = BAR
        x -= 1


def transition_function(state, action, frame):
    out = [list(map(int, r)) for r in frame]
    cont = _mem["frame"] is not None and _mem["frame"] == frame
    players = find(state, lambda o: o.get("type") == "player")
    bars = find(state, lambda o: o.get("type") == "counter" and has_tag(o, "budget"))
    glyphs = find(state, lambda o: o.get("type") == "counter" and has_tag(o, "progress"))
    if not players:
        _mem.update(frame=out, under=None)
        return out
    p = players[0]
    px, py, s = p["x"], p["y"], p["w"]
    under = _mem["under"] if cont and _mem["under"] else [[FLOOR] * s for _ in range(s)]
    aid = action if isinstance(action, int) else action.get("action_id")
    nx, ny = px, py
    if aid in DIRS:
        dx, dy = DIRS[aid]
        if cell_ok(frame, px + dx * s, py + dy * s, s):
            nx, ny = px + dx * s, py + dy * s
            d = belt_push(frame, nx, ny, s)
            while d and cell_ok(frame, nx + d[0] * s, ny + d[1] * s, s):
                nx, ny = nx + d[0] * s, ny + d[1] * s
    if (nx, ny) != (px, py):
        for lg in find(state, lambda o: has_tag(o, "chamber")):
            if overlaps(lg, nx, ny, s) and glyphs:
                want = legend_info(frame, lg)
                if glyph_mask(frame, glyphs[0]) != want:
                    _mem.update(frame=out, under=under)
                    return out
    for b in bars:
        burn(out, b)
    if (nx, ny) != (px, py):
        for j in range(s):
            for i in range(s):
                out[py + j][px + i] = under[j][i]
        new_under = [[frame[ny + j][nx + i] for i in range(s)] for j in range(s)]
        for j in range(s):
            for i in range(s):
                if px <= nx + i < px + s and py <= ny + j < py + s:
                    new_under[j][i] = under[ny + j - py][nx + i - px]
        for o in state:
            if o is p or not covered(o, nx, ny, s):
                continue
            if o.get("type") == "refuel":
                for b in bars:
                    refill(out, b)
            elif has_tag(o, "collect"):
                for g in glyphs:
                    rotate_glyph(out, frame, g)
                continue
            elif has_tag(o, "recolor"):
                for g in glyphs:
                    recolor_glyph(out, g, 9)
            else:
                continue
            for j in range(o["h"]):
                for i in range(o["w"]):
                    new_under[o["y"] + j - ny][o["x"] + i - nx] = FLOOR
        for j in range(s):
            for i in range(s):
                out[ny + j][nx + i] = p["pixels"][j][i]
        under = new_under
    _mem.update(frame=[r[:] for r in out], under=under)
    return out
