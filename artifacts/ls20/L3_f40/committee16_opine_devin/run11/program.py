# Mechanics: player 5x5 moves 5 cells (A1-4); dest box containing wall colour 4 / void 5 / off-board = blocked (still costs).
# A colour-1 strip on an edge of the dest cell is a conveyor: player slides away from it until blocked.
# Every action burns the leftmost 2 bar cells (colour 11 -> 3, rows 61-62); a refuel ring fully covered is consumed + bar refilled.
# Covering a 'collect' target rotates the 3x3 HUD glyph (2x2 blocks) 90deg CW; covered cells are remembered (continuity-gated) and restored.
# Unconfirmed: dest overlapping a 'chamber' object while glyph != legend is a free no-op; unlock condition and A5-A7 unobserved.
WALL, VOID, FLOOR, STRIP, BAR = 4, 5, 3, 1, 11
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"frame": None, "under": None}


def box(x, y, w, h):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def inside(o, x, y, s=5):
    return x <= o["x"] and y <= o["y"] and o["x"] + o["w"] <= x + s and o["y"] + o["h"] <= y + s


def overlaps(o, x, y, s=5):
    return o["x"] < x + s and x < o["x"] + o["w"] and o["y"] < y + s and y < o["y"] + o["h"]


def passable(frame, x, y):
    for cx, cy in box(x, y, 5, 5):
        if not (0 <= cx < 64 and 0 <= cy < 60) or frame[cy][cx] in (WALL, VOID):
            return False
    return True


def conveyor(frame, x, y):
    """Direction the strip on an edge of cell (x,y) pushes, or None."""
    def strip(cells):
        return all(0 <= cx < 64 and 0 <= cy < 64 and frame[cy][cx] == STRIP for cx, cy in cells)
    if strip([(x - 1, y + j) for j in range(5)]):
        return (1, 0)
    if strip([(x + 5, y + j) for j in range(5)]):
        return (-1, 0)
    if strip([(x + i, y - 1) for i in range(5)]):
        return (0, 1)
    if strip([(x + i, y + 5) for i in range(5)]):
        return (0, -1)
    return None


def glyph_grid(frame, g):
    cells = {}
    for r in range(3):
        for c in range(3):
            cells[(r, c)] = frame[g["y"] + 2 * r][g["x"] + 2 * c]
    return cells


def rotate_glyph(out, frame, g):
    old = glyph_grid(frame, g)
    for r in range(3):
        for c in range(3):
            v = old[(2 - c, r)]
            for dy in range(2):
                for dx in range(2):
                    out[g["y"] + 2 * r + dy][g["x"] + 2 * c + dx] = v


def glyph_matches_legend(frame, g, legend):
    pat = next((t for t in legend["tags"] if len(t) == 9 and set(t) <= set("#.")), None)
    if pat is None:
        return False
    cells = glyph_grid(frame, g)
    shape = "".join("#" if cells[(r, c)] != VOID else "." for r in range(3) for c in range(3))
    colours = {v for v in cells.values() if v != VOID}
    lc = {frame[y][x] for x, y in box(legend["x"], legend["y"], legend["w"], legend["h"])} - {VOID}
    return shape == pat and colours <= lc


def burn_bar(out, frame, refill):
    for row in (61, 62):
        track = [x for x in range(64) if frame[row][x] in (BAR, FLOOR)]
        if refill:
            for x in track:
                out[row][x] = BAR
            continue
        lit = [x for x in track if frame[row][x] == BAR]
        for x in lit[:2]:
            out[row][x] = FLOOR


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    player = next((o for o in state if o["type"] == "player"), None)
    under = _mem["under"] if _mem["frame"] == frame else None
    if player is None:
        return out
    px, py = player["x"], player["y"]
    if under is None:
        under = {c: FLOOR for c in box(px, py, 5, 5)}
    aid = action if isinstance(action, int) else action.get("action_id")
    nx, ny = px, py
    if aid in MOVES:
        dx, dy = MOVES[aid]
        tx, ty = px + 5 * dx, py + 5 * dy
        if passable(frame, tx, ty):
            nx, ny = tx, ty
            push = conveyor(frame, nx, ny)
            if push:
                while passable(frame, nx + 5 * push[0], ny + 5 * push[1]):
                    nx, ny = nx + 5 * push[0], ny + 5 * push[1]
    glyph = next((o for o in state if o["type"] == "counter" and "progress" in o["tags"]), None)
    if (nx, ny) != (px, py):
        for o in state:
            if "chamber" in o["tags"] and overlaps(o, nx, ny):
                if not (glyph and glyph_matches_legend(frame, glyph, o)):
                    return out
    refill = False
    if (nx, ny) != (px, py):
        new_under = {c: frame[c[1]][c[0]] for c in box(nx, ny, 5, 5)}
        for o in state:
            if o["type"] == "player" or not inside(o, nx, ny):
                continue
            if o["type"] == "refuel":
                refill = True
                for c in box(o["x"], o["y"], o["w"], o["h"]):
                    new_under[c] = FLOOR
            if "collect" in o["tags"] and glyph:
                rotate_glyph(out, frame, glyph)
        for (cx, cy), v in under.items():
            out[cy][cx] = v
        under = new_under
        pix = player.get("pixels") or [[12] * 5] * 5
        for j in range(5):
            for i in range(5):
                out[ny + j][nx + i] = pix[j][i]
    burn_bar(out, frame, refill)
    _mem["frame"], _mem["under"] = out, under
    return [list(r) for r in out]

