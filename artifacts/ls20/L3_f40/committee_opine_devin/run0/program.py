# Mechanics: player = 5x5 sprite moving 5 cells (A1 up, A2 down, A3 left, A4 right); a move is blocked when the
# destination box leaves the board or holds wall colour 4. A destination with a full colour-1 strip on an edge is a
# conveyor: the player slides away from the strip until blocked. Every action burns 2 cells (11 -> 3) from the left of
# the step bar (rows of an 11/3 run); a fully covered refuel ring is consumed and refills the bar instead. A fully
# covered 'collect' target rotates the 3x3 HUD glyph 90deg CW and is restored when uncovered (continuity-gated cache).
# Unconfirmed (unobserved here): recolor button -> glyph colour 9, chamber refusal (free) while glyph != legend.
WALL, FLOOR, BAR, BG5, CONVEYOR = 4, 3, 11, 5, 1
MOVES = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
_mem = {"frame": None, "hidden": []}


def box(o):
    return o["x"], o["y"], o["w"], o["h"]


def covers(px, py, pw, ph, o):
    x, y, w, h = box(o)
    return px <= x and py <= y and x + w <= px + pw and y + h <= py + ph


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def blocked(frame, x, y, w, h):
    if x < 0 or y < 0 or x + w > 64 or y + h > 64:
        return True
    return any(frame[yy][xx] == WALL for yy in range(y, y + h) for xx in range(x, x + w))


def conveyor_dir(frame, x, y, w, h):
    def strip(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and frame[cy][cx] == CONVEYOR for cx, cy in cells)
    if strip([(x - 1, y + i) for i in range(h)]):
        return (w, 0)
    if strip([(x + w, y + i) for i in range(h)]):
        return (-w, 0)
    if strip([(x + i, y - 1) for i in range(w)]):
        return (0, h)
    if strip([(x + i, y + h) for i in range(w)]):
        return (0, -h)
    return None


def bar_track(frame, obj):
    x, y, w, h = box(obj)
    row = frame[y]
    lo, hi = x, x + w - 1
    while lo - 1 >= 0 and row[lo - 1] in (BAR, FLOOR):
        lo -= 1
    while hi + 1 < 64 and row[hi + 1] in (BAR, FLOOR):
        hi += 1
    return lo, hi, y, h


def glyph_cells(frame, g):
    x, y, w, h = box(g)
    n, s = 3, w // 3
    mask = [[frame[y + r * s][x + c * s] != BG5 for c in range(n)] for r in range(n)]
    cols = [frame[y + r * s][x + c * s] for r in range(n) for c in range(n) if mask[r][c]]
    return mask, (cols[0] if cols else BG5), s


def draw_glyph(out, g, mask, colour, s):
    x, y = g["x"], g["y"]
    for r in range(3):
        for c in range(3):
            v = colour if mask[r][c] else BG5
            for dy in range(s):
                for dx in range(s):
                    out[y + r * s + dy][x + c * s + dx] = v


def rotate_cw(m):
    n = len(m)
    return [[m[n - 1 - j][i] for j in range(n)] for i in range(n)]


def legend_info(frame, leg):
    pat = [t for t in leg.get("tags", []) if len(t) == 9 and set(t) <= set("#.")]
    x, y, w, h = box(leg)
    cols = [frame[yy][xx] for yy in range(y, y + h) for xx in range(x, x + w) if frame[yy][xx] != BG5]
    mask = [[ch == "#" for ch in pat[0][r * 3:r * 3 + 3]] for r in range(3)] if pat else None
    return mask, (cols[0] if cols else None)


def chamber_refuses(frame, dest, objs, glyph):
    for o in objs:
        if "chamber" not in o.get("tags", []):
            continue
        x, y, w, h = box(o)
        if not overlaps(dest, (x - 3, y - 3, w + 6, h + 6)):
            continue
        if glyph is None:
            return True
        mask, colour, _ = glyph_cells(frame, glyph)
        lmask, lcol = legend_info(frame, o)
        if mask != lmask or colour != lcol:
            return True
    return False


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    hidden = _mem["hidden"] if _mem["frame"] == frame else []
    hidden = [h for h in hidden if not any(o.get("name") == h[0] for o in state)]
    player = next((o for o in state if o.get("type") == "player"), None)
    glyph = next((o for o in state if o.get("type") == "counter" and "progress" in o.get("tags", [])), None)
    bar = next((o for o in state if o.get("type") == "counter" and "budget" in o.get("tags", [])), None)
    if player is None:
        return out
    px, py, pw, ph = box(player)
    nx, ny = px, py
    aid = action if isinstance(action, int) else action.get("action_id")
    if aid in MOVES:
        dx, dy = MOVES[aid]
        tx, ty = px + dx, py + dy
        if not blocked(frame, tx, ty, pw, ph):
            if chamber_refuses(frame, (tx, ty, pw, ph), state, glyph):
                _mem["frame"], _mem["hidden"] = out, hidden
                return out
            nx, ny = tx, ty
            d = conveyor_dir(frame, nx, ny, pw, ph)
            if d:
                while not blocked(frame, nx + d[0], ny + d[1], pw, ph):
                    nx, ny = nx + d[0], ny + d[1]
    refuelled = False
    if (nx, ny) != (px, py):
        for yy in range(py, py + ph):
            for xx in range(px, px + pw):
                out[yy][xx] = FLOOR
        keep = []
        for name, (x, y, w, h), pix in hidden:
            if overlaps((nx, ny, pw, ph), (x, y, w, h)):
                keep.append((name, (x, y, w, h), pix))
                continue
            for r in range(h):
                for c in range(w):
                    out[y + r][x + c] = pix[r][c]
        hidden = keep
        for o in state:
            if o is player or not covers(nx, ny, pw, ph, o):
                continue
            tags = o.get("tags", [])
            if o.get("type") == "refuel":
                refuelled = True
            elif "collect" in tags:
                x, y, w, h = box(o)
                hidden.append((o["name"], (x, y, w, h), [frame[y + r][x:x + w] for r in range(h)]))
                if glyph is not None:
                    mask, colour, s = glyph_cells(out, glyph)
                    draw_glyph(out, glyph, rotate_cw(mask), colour, s)
            elif "recolor" in tags and glyph is not None:
                mask, colour, s = glyph_cells(out, glyph)
                draw_glyph(out, glyph, mask, 9, s)
        for yy in range(ny, ny + ph):
            for xx in range(nx, nx + pw):
                out[yy][xx] = player["pixels"][yy - ny][xx - nx]
    if bar is not None:
        lo, hi, by, bh = bar_track(frame, bar)
        for yy in range(by, by + bh):
            if refuelled:
                for xx in range(lo, hi + 1):
                    out[yy][xx] = BAR
            else:
                left = [xx for xx in range(lo, hi + 1) if out[yy][xx] == BAR][:2]
                for xx in left:
                    out[yy][xx] = FLOOR
    _mem["frame"], _mem["hidden"] = out, hidden
    return out
