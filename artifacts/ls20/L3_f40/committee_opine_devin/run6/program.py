# Mechanics: player = 5x5 block stepping 5 cells (A1-4); a move is blocked when the destination box leaves the board or holds
# wall colour 4 / void colour 5 (blocked moves still burn). A full 5-cell colour-1 strip on a destination edge is a conveyor
# that slides the player away from it until blocked. Every action burns the 2 leftmost 11-cells of the bar (rows 61-62);
# a refuel ring inside the landing box is consumed and refills the bar. A 'collect' sprite inside the box rotates the 3x3
# HUD glyph 90deg CW and survives under the player (continuity-gated cache); chamber entry refused free unless glyph==legend.
WALL, VOID, FLOOR, BAR, CONVEYOR = 4, 5, 3, 11, 1
DIRS = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
_mem = {"out": None, "under": None}


def obj(state, typ=None, tag=None):
    for o in state:
        if (typ is None or o.get("type") == typ) and (tag is None or tag in o.get("tags", [])):
            return o
    return None


def objs(state, tag):
    return [o for o in state if tag in o.get("tags", [])]


def box_cells(x, y, w, h):
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def blocked(frame, x, y):
    if x < 0 or y < 0 or x + 5 > 64 or y + 5 > 64:
        return True
    return any(frame[j][i] in (WALL, VOID) for i, j in box_cells(x, y, 5, 5))


def conveyor_dir(frame, x, y):
    def strip(cells):
        return all(0 <= i < 64 and 0 <= j < 64 and frame[j][i] == CONVEYOR for i, j in cells)
    if strip([(x - 1, y + k) for k in range(5)]): return (5, 0)
    if strip([(x + 5, y + k) for k in range(5)]): return (-5, 0)
    if strip([(x + k, y - 1) for k in range(5)]): return (0, 5)
    if strip([(x + k, y + 5) for k in range(5)]): return (0, -5)
    return None


def inside(o, x, y):
    return o["x"] >= x and o["y"] >= y and o["x"] + o["w"] <= x + 5 and o["y"] + o["h"] <= y + 5


def overlaps(o, x, y):
    return o["x"] < x + 5 and x < o["x"] + o["w"] and o["y"] < y + 5 and y < o["y"] + o["h"]


def sprite_cells(frame, o):
    """Object bbox grown over 8-adjacent non-{floor,wall,void} cells."""
    seen = set(box_cells(o["x"], o["y"], o["w"], o["h"]))
    todo = list(seen)
    while todo:
        i, j = todo.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                c = (i + di, j + dj)
                if c not in seen and 0 <= c[0] < 64 and 0 <= c[1] < 64 and frame[c[1]][c[0]] not in (FLOOR, WALL, VOID):
                    seen.add(c); todo.append(c)
    return seen


def glyph_read(frame, g):
    mask = [[frame[g["y"] + 2 * r][g["x"] + 2 * c] != VOID for c in range(3)] for r in range(3)]
    cols = [frame[g["y"] + 2 * r][g["x"] + 2 * c] for r in range(3) for c in range(3) if mask[r][c]]
    return mask, (cols[0] if cols else None)


def glyph_write(out, g, mask, colour):
    for r in range(3):
        for c in range(3):
            for i, j in box_cells(g["x"] + 2 * c, g["y"] + 2 * r, 2, 2):
                out[j][i] = colour if mask[r][c] else VOID


def legend_read(frame, lg):
    pat = next((t for t in lg["tags"] if len(t) == 9 and set(t) <= set("#.")), None)
    mask = [[pat[3 * r + c] == "#" for c in range(3)] for r in range(3)] if pat else None
    cols = [frame[j][i] for i, j in box_cells(lg["x"], lg["y"], lg["w"], lg["h"]) if frame[j][i] != VOID]
    return mask, (cols[0] if cols else None)


def bar_track(frame):
    row = 61
    xs = [x for x in range(64) if frame[row][x] == BAR]
    if not xs:
        return row, []
    lo, hi = xs[0], xs[-1]
    while lo - 1 >= 0 and frame[row][lo - 1] in (BAR, FLOOR): lo -= 1
    while hi + 1 < 64 and frame[row][hi + 1] in (BAR, FLOOR): hi += 1
    return row, list(range(lo, hi + 1))


def track_bounds(frame):
    row = 61
    xs = [x for x in range(64) if frame[row][x] in (BAR,)]
    if xs:
        return bar_track(frame)
    # empty bar: track = floor run right of the HUD delimiter
    run = [x for x in range(13, 64) if frame[row][x] == FLOOR]
    return row, run


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    under = _mem["under"] if _mem["out"] == frame else None
    p = obj(state, "player")
    g = obj(state, "counter", "progress")
    lg = obj(state, None, "chamber")
    if isinstance(action, dict) or p is None:
        return out
    px, py = p["x"], p["y"]
    nx, ny = px, py
    if action in DIRS:
        dx, dy = DIRS[action]
        if not blocked(frame, px + dx, py + dy):
            nx, ny = px + dx, py + dy
            cd = conveyor_dir(frame, nx, ny)
            if cd:
                while not blocked(frame, nx + cd[0], ny + cd[1]):
                    nx, ny = nx + cd[0], ny + cd[1]
    moved = (nx, ny) != (px, py)
    if moved and lg is not None and overlaps(lg, nx, ny) and g is not None:
        if glyph_read(frame, g) != legend_read(frame, lg):
            _mem["out"], _mem["under"] = out, under
            return out
    # burn 2 budget cells
    row, track = track_bounds(frame)
    lit = [x for x in track if frame[row][x] == BAR]
    for x in lit[:2]:
        out[row][x] = out[row + 1][x] = FLOOR
    if moved:
        player_px = [[frame[py + j][px + i] for i in range(5)] for j in range(5)]
        for i, j in box_cells(px, py, 5, 5):
            out[j][i] = under.get((i, j), FLOOR) if under else FLOOR
        new_under = {(i, j): out[j][i] for i, j in box_cells(nx, ny, 5, 5)}
        for o in objs(state, "budget"):
            if o.get("type") == "refuel" and inside(o, nx, ny):
                for c in sprite_cells(frame, o):
                    new_under[c] = FLOOR
                    out[c[1]][c[0]] = FLOOR
                for x in track:
                    out[row][x] = out[row + 1][x] = BAR
        for o in objs(state, "recolor"):
            if inside(o, nx, ny):
                for c in sprite_cells(frame, o):
                    new_under[c] = FLOOR
                    out[c[1]][c[0]] = FLOOR
                if g is not None:
                    mask, _ = glyph_read(out, g)
                    glyph_write(out, g, mask, legend_read(frame, lg)[1] if lg else 9)
        for o in objs(state, "collect"):
            if inside(o, nx, ny) and g is not None:
                mask, colour = glyph_read(out, g)
                glyph_write(out, g, [[mask[2 - c][r] for c in range(3)] for r in range(3)], colour)
        for i, j in box_cells(nx, ny, 5, 5):
            out[j][i] = player_px[j - ny][i - nx]
        under = new_under
    _mem["out"], _mem["under"] = out, under
    return out
