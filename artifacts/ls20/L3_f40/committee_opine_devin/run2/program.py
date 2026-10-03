# Mechanics: player 5x5 sprite moves 5 cells (A1-4); dest box containing wall colour 4 or leaving the board = blocked.
# A full 5-cell colour-1 strip on a landed cell's edge is a conveyor: slide away from it until blocked.
# Every action burns 2 bar cells (colour 11 -> 3, from the left); a refuel ring fully covered is consumed and refills the bar.
# A 'collect' object fully covered rotates the 3x3 HUD glyph 90deg CW; it is occluded (restored from a continuity-gated cache).
# Unconfirmed: recolor token (glyph -> 9) and chamber guard (dest near legend while glyph != legend -> free no-op).
WALL, FLOOR, CONVEYOR, BAR, VOID = 4, 3, 1, 11, 5
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"frame": None, "cache": {}}


def box_cells(x, y, w, h):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def inside(outer, inner):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ix + iw <= ox + ow and iy + ih <= oy + oh


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def bbox(o):
    return (o["x"], o["y"], o["w"], o["h"])


def passable(frame, x, y, w, h, player_box):
    for cx, cy in box_cells(x, y, w, h):
        if not (0 <= cx < 64 and 0 <= cy < 64):
            return False
        if frame[cy][cx] == WALL and not overlaps((cx, cy, 1, 1), player_box):
            return False
    return True


def conveyor_dir(frame, x, y, w, h):
    def strip(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and frame[cy][cx] == CONVEYOR for cx, cy in cells)
    if strip([(x - 1, y + j) for j in range(h)]):
        return (1, 0)
    if strip([(x + w, y + j) for j in range(h)]):
        return (-1, 0)
    if strip([(x + i, y - 1) for i in range(w)]):
        return (0, 1)
    if strip([(x + i, y + h) for i in range(w)]):
        return (0, -1)
    return None


def glyph_read(frame, g):
    mask, colour = [], None
    for r in range(3):
        row = []
        for c in range(3):
            v = frame[g["y"] + 2 * r][g["x"] + 2 * c]
            row.append(v != VOID)
            if v != VOID:
                colour = v
        mask.append(row)
    return mask, colour


def glyph_write(out, g, mask, colour):
    for r in range(3):
        for c in range(3):
            for dy in range(2):
                for dx in range(2):
                    out[g["y"] + 2 * r + dy][g["x"] + 2 * c + dx] = colour if mask[r][c] else VOID


def rotate_cw(mask):
    return [[mask[2 - c][r] for c in range(3)] for r in range(3)]


def legend_mask(o):
    pat = next((t for t in o["tags"] if len(t) == 9 and set(t) <= set("#.")), None)
    if pat is None:
        return None
    return [[pat[3 * r + c] == "#" for c in range(3)] for r in range(3)]


def bar_track(frame):
    y = 61
    runs, start = [], None
    for x in range(65):
        v = frame[y][x] if x < 64 else VOID
        if v in (BAR, FLOOR):
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x - 1))
            start = None
    runs = [r for r in runs if any(frame[y][x] == BAR for x in range(r[0], r[1] + 1))] or runs
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def burn_bar(out, frame, refill):
    tr = bar_track(frame)
    if tr is None:
        return
    rows = [61, 62]
    cols = list(range(tr[0], tr[1] + 1))
    if refill:
        for y in rows:
            for x in cols:
                out[y][x] = BAR
        return
    left = [x for x in cols if frame[61][x] == BAR][:2]
    for y in rows:
        for x in left:
            out[y][x] = FLOOR


def sprite(frame, o):
    return [[frame[o["y"] + j][o["x"] + i] for i in range(o["w"])] for j in range(o["h"])]


def paint(out, x, y, px):
    for j, row in enumerate(px):
        for i, v in enumerate(row):
            if 0 <= y + j < 64 and 0 <= x + i < 64 and v >= 0:
                out[y + j][x + i] = v


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    continuous = _mem["frame"] is not None and _mem["frame"] == frame
    if not continuous:
        _mem["cache"] = {}
    names = {o["name"] for o in state}
    for o in state:
        if o["type"] != "player" and o["layer"] == 0 and o["y"] < 51:
            _mem["cache"][o["name"]] = dict(o, sprite=sprite(frame, o))
    hidden = [c for n, c in _mem["cache"].items() if n not in names]
    player = next((o for o in state if o["type"] == "player"), None)
    glyph = next((o for o in state if o["name"] == "hud_glyph" or "progress" in o["tags"]), None)
    legend = next((o for o in state if "chamber" in o["tags"]), None)
    if player is None or not isinstance(action, int) or action not in DIRS:
        if isinstance(action, int) and action in (5, 7) or not isinstance(action, int):
            burn_bar(out, frame, False)
        _mem["frame"] = out
        return out
    pbox = bbox(player)
    dx, dy = DIRS[action]
    nx, ny = player["x"] + dx * 5, player["y"] + dy * 5
    w, h = player["w"], player["h"]
    if legend is not None and glyph is not None:
        lx, ly, lw, lh = bbox(legend)
        if overlaps((nx, ny, w, h), (lx - 3, ly - 3, lw + 6, lh + 6)):
            if glyph_read(frame, glyph) != (legend_mask(legend), legend["pixels"][0][0] if "pixels" in legend else 9):
                _mem["frame"] = out
                return out
    moved = passable(frame, nx, ny, w, h, pbox)
    if moved:
        conv = conveyor_dir(frame, nx, ny, w, h)
        if conv is not None:
            while passable(frame, nx + conv[0] * 5, ny + conv[1] * 5, w, h, pbox):
                nx, ny = nx + conv[0] * 5, ny + conv[1] * 5
    refill = False
    if moved:
        dest = (nx, ny, w, h)
        for cx, cy in box_cells(*pbox):
            out[cy][cx] = FLOOR
        for c in hidden:
            if not inside(dest, bbox(c)):
                paint(out, c["x"], c["y"], c["sprite"])
        gmask, gcol = glyph_read(frame, glyph) if glyph is not None else (None, None)
        for o in state:
            if o is player or o["layer"] != 0 or not inside(dest, bbox(o)):
                continue
            if o["type"] == "refuel":
                refill = True
                _mem["cache"].pop(o["name"], None)
            elif "collect" in o["tags"] and gmask is not None:
                gmask = rotate_cw(gmask)
            elif "recolor" in o["tags"] and gmask is not None:
                gcol = 9
                _mem["cache"].pop(o["name"], None)
        if glyph is not None:
            glyph_write(out, glyph, gmask, gcol)
        paint(out, nx, ny, player.get("pixels") or [[frame[player["y"] + j][player["x"] + i] for i in range(w)] for j in range(h)])
    burn_bar(out, frame, refill)
    _mem["frame"] = out
    return out
