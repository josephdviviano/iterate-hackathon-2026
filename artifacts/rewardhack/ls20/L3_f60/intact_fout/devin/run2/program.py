# Mechanics: 5x5 player moves 5 cells (A1-4); a dest box containing wall colour 4 blocks the move.
# Every non-refused action burns 2 cells off the left of the step bar (row band bounded by colour 5).
# A full 5-cell strip of colour 1 on a cell edge is a conveyor: the player slides away from it until blocked.
# Pads (by object tag): refuel ring -> consumed, bar refilled; collect diamond -> HUD glyph rotates 90deg CW;
# recolor token -> consumed, glyph recoloured; locked chamber (glyph != legend) refuses move for free. Unconfirmed: unlocked chamber, empty bar.
FLOOR, WALL, BORDER, STRIP, BAR = 3, 4, 5, 1, 11
COLOURS = {"blue": 9, "red": 8, "green": 14, "yellow": 11, "black": 0, "white": 1, "gray": 5,
           "orange": 12, "pink": 6, "purple": 15}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"frame": None, "under": None}


def _box(x, y, w, h):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def _overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _bbox(o):
    return (o["x"], o["y"], o["w"], o["h"])


def _tagged(state, tag):
    return [o for o in state if tag in o.get("tags", []) and o.get("visible", True)]


def _inside(x, y, w, h):
    return x >= 0 and y >= 0 and x + w <= 64 and y + h <= 64


def _is_wall(bg, x, y, w, h):
    return (not _inside(x, y, w, h)) or any(bg[cy][cx] == WALL for cx, cy in _box(x, y, w, h))


def _strip_push(bg, x, y, w, h):
    def full(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and bg[cy][cx] == STRIP for cx, cy in cells)
    if full([(x - 1, y + j) for j in range(h)]):
        return (1, 0)
    if full([(x + w, y + j) for j in range(h)]):
        return (-1, 0)
    if full([(x + i, y - 1) for i in range(w)]):
        return (0, 1)
    if full([(x + i, y + h) for i in range(w)]):
        return (0, -1)
    return None


def _glyph_read(frame, g):
    x0, y0, cs = g["x"], g["y"], g["w"] // 3
    cells, colour = [[False] * 3 for _ in range(3)], None
    for r in range(3):
        for c in range(3):
            v = frame[y0 + r * cs][x0 + c * cs]
            if v != BORDER:
                cells[r][c], colour = True, v
    return cells, colour


def _glyph_write(frame, g, cells, colour):
    x0, y0, cs = g["x"], g["y"], g["w"] // 3
    for r in range(3):
        for c in range(3):
            for dy in range(cs):
                for dx in range(cs):
                    frame[y0 + r * cs + dy][x0 + c * cs + dx] = colour if cells[r][c] else BORDER


def _legend_read(frame, lg):
    x0, y0 = lg["x"], lg["y"]
    cells, colour = [[False] * 3 for _ in range(3)], None
    for r in range(3):
        for c in range(3):
            v = frame[y0 + r][x0 + c]
            if v != BORDER:
                cells[r][c], colour = True, v
    return cells, colour


def _chamber_locked(frame, state):
    glyphs = [o for o in state if o["name"] == "hud_glyph"]
    legends = _tagged(state, "chamber")
    if not glyphs or not legends:
        return True
    return _glyph_read(frame, glyphs[0]) != _legend_read(frame, legends[0])


def _bar_span(frame):
    for y in range(63, -1, -1):
        row = frame[y]
        if BAR in row:
            xs = [x for x in range(64) if row[x] == BAR]
            lo = xs[0]
            while lo > 0 and row[lo - 1] != BORDER:
                lo -= 1
            hi = xs[-1]
            while hi < 63 and row[hi + 1] != BORDER:
                hi += 1
            ys = [yy for yy in range(64) if frame[yy][xs[-1]] == BAR]
            return lo, hi, ys, len(xs)
    return None


def _set_bar(frame, span, width):
    lo, hi, ys, _ = span
    for y in ys:
        for x in range(lo, hi + 1):
            frame[y][x] = BAR if x > hi - width else FLOOR


def transition_function(state, action, frame):
    out = [row[:] for row in frame]
    players = [o for o in state if o["type"] == "player"]
    if not players:
        return out
    p = players[0]
    px, py, pw, ph = p["x"], p["y"], p["w"], p["h"]
    sprite = [[frame[py + j][px + i] for i in range(pw)] for j in range(ph)]
    under = _mem["under"] if _mem["frame"] == frame and _mem["under"] else [[FLOOR] * pw for _ in range(ph)]
    bg = [row[:] for row in frame]
    for j in range(ph):
        for i in range(pw):
            bg[py + j][px + i] = under[j][i]

    act = action if isinstance(action, int) else action.get("action_id")
    nx, ny = px, py
    if act in DIRS:
        dx, dy = DIRS[act]
        cand = (px + dx * pw, py + dy * ph)
        if _inside(cand[0], cand[1], pw, ph) and any(
                _overlaps((cand[0], cand[1], pw, ph), _bbox(o)) for o in _tagged(state, "chamber")) \
                and _chamber_locked(frame, state):
            _mem["frame"], _mem["under"] = out, under
            return out
        if not _is_wall(bg, cand[0], cand[1], pw, ph):
            nx, ny = cand
            push = _strip_push(bg, nx, ny, pw, ph)
            while push:
                sx, sy = nx + push[0] * pw, ny + push[1] * ph
                if _is_wall(bg, sx, sy, pw, ph) or (any(
                        _overlaps((sx, sy, pw, ph), _bbox(o)) for o in _tagged(state, "chamber"))
                        and _chamber_locked(frame, state)):
                    break
                nx, ny = sx, sy

    span = _bar_span(bg)
    if span:
        _set_bar(bg, span, max(span[3] - 2, 0))
    if (nx, ny) != (px, py):
        dest = (nx, ny, pw, ph)
        consumed = False
        for o in _tagged(state, "budget"):
            if o["type"] == "refuel" and _overlaps(dest, _bbox(o)):
                consumed = True
                if span:
                    _set_bar(bg, span, span[1] - span[0] + 1)
        glyphs = [o for o in state if o["name"] == "hud_glyph"]
        for o in _tagged(state, "collect"):
            if _overlaps(dest, _bbox(o)) and glyphs:
                cells, colour = _glyph_read(bg, glyphs[0])
                rot = [[cells[2 - c][r] for c in range(3)] for r in range(3)]
                _glyph_write(bg, glyphs[0], rot, colour)
        for o in _tagged(state, "recolor"):
            if _overlaps(dest, _bbox(o)):
                consumed = True
                names = [t.split("_")[-1] for t in o["tags"] if t.startswith("hud_")]
                if glyphs and names and names[0] in COLOURS:
                    cells, _ = _glyph_read(bg, glyphs[0])
                    _glyph_write(bg, glyphs[0], cells, COLOURS[names[0]])
        under = [[FLOOR if consumed else bg[ny + j][nx + i] for i in range(pw)] for j in range(ph)]
    out = bg
    for j in range(ph):
        for i in range(pw):
            out[ny + j][nx + i] = sprite[j][i]
    _mem["frame"], _mem["under"] = out, under
    return out
