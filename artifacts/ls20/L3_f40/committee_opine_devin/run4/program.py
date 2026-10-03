# Mechanics: player 5x5 moves 5 cells (A1-4); dest blocked if off-board or any wall (4) / void (5) cell,
# a 5-cell colour-1 strip on the dest's edge is a conveyor sliding the player away until blocked.
# step_bar (rows y,y+1 of its object) burns 2 cells (11->3) from the left per action, incl. blocked moves;
# covering a refuel ring consumes it and refills the bar; covering a 'collect' target rotates the HUD glyph
# 90deg CW and leaves it under the player (restored via continuity-gated memory). Unconfirmed: chamber unlock.
FLOOR, WALL, VOID, CONV, BAR = 3, 4, 5, 1, 11
DIRS = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
_mem = {"frame": None, "under": None}


def find(state, pred):
    return [o for o in state if pred(o)]


def contains(box, o):
    x, y, w, h = box
    return x <= o["x"] and y <= o["y"] and o["x"] + o["w"] <= x + w and o["y"] + o["h"] <= y + h


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(box):
    x, y, w, h = box
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def in_board(box):
    return box[0] >= 0 and box[1] >= 0 and box[0] + box[2] <= 64 and box[1] + box[3] <= 64


def wall_blocked(frame, box, own):
    if not in_board(box):
        return True
    return any((c not in own) and frame[c[1]][c[0]] in (WALL, VOID) for c in cells(box))


def conveyor_dir(frame, box):
    x, y, w, h = box
    def strip(pts):
        return all(0 <= px < 64 and 0 <= py < 64 and frame[py][px] == CONV for px, py in pts)
    if strip([(x - 1, y + j) for j in range(h)]):
        return (w, 0)
    if strip([(x + w, y + j) for j in range(h)]):
        return (-w, 0)
    if strip([(x + i, y - 1) for i in range(w)]):
        return (0, h)
    if strip([(x + i, y + h) for i in range(w)]):
        return (0, -h)
    return None


def glyph_mask(frame, g):
    m, col = [], None
    for r in range(3):
        row = []
        for c in range(3):
            v = frame[g["y"] + 2 * r][g["x"] + 2 * c]
            row.append(v != VOID)
            if v != VOID:
                col = v
        m.append(row)
    return m, col


def draw_glyph(out, g, m, col):
    for r in range(3):
        for c in range(3):
            for dy in range(2):
                for dx in range(2):
                    out[g["y"] + 2 * r + dy][g["x"] + 2 * c + dx] = col if m[r][c] else VOID


def rotate_cw(m):
    return [[m[2 - j][i] for j in range(3)] for i in range(3)]


def chamber_locked(frame, state):
    g = find(state, lambda o: o["name"] == "hud_glyph")
    legend = find(state, lambda o: "chamber" in o.get("tags", []))
    if not g or not legend:
        return None
    lg = legend[0]
    pat = [t for t in lg["tags"] if set(t) <= {"#", "."} and len(t) == 9]
    m, col = glyph_mask(frame, g[0])
    want = [[pat[0][3 * r + c] == "#" for c in range(3)] for r in range(3)] if pat else m
    lcol = next((frame[y][x] for x, y in cells((lg["x"], lg["y"], lg["w"], lg["h"]))
                 if frame[y][x] not in (VOID, FLOOR)), col)
    locked = m != want or col != lcol
    zone = (lg["x"] - 3, lg["y"] - 3, lg["w"] + 6, lg["h"] + 6)
    return zone if locked else None


def burn(out, bar):
    y = bar["y"]
    x0 = bar["x"]
    while x0 > 0 and out[y][x0 - 1] in (BAR, FLOOR):
        x0 -= 1
    run = []
    for x in range(x0, 64):
        if out[y][x] in (BAR, FLOOR):
            run.append(x)
        else:
            break
    full = [x for x in run if out[y][x] == BAR][:2]
    for x in full:
        for dy in range(bar["h"]):
            out[y + dy][x] = FLOOR
    return run


def refill(out, bar, run):
    for x in run:
        for dy in range(bar["h"]):
            out[bar["y"] + dy][x] = BAR


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    cont = _mem["frame"] is not None and _mem["frame"] == frame
    under = _mem["under"] if cont else None
    players = find(state, lambda o: o["type"] == "player")
    bars = find(state, lambda o: o["name"] == "step_bar" or (o["type"] == "counter" and "budget" in o.get("tags", [])))
    if not players or not isinstance(action, int) or action not in DIRS:
        if bars:
            burn(out, bars[0])
        _mem["frame"], _mem["under"] = out, under
        return out
    p = players[0]
    box = (p["x"], p["y"], p["w"], p["h"])
    own = set(cells(box))
    dx, dy = DIRS[action]
    dest = (box[0] + dx, box[1] + dy, box[2], box[3])
    zone = chamber_locked(frame, state)
    if zone and overlaps(dest, zone):
        _mem["frame"], _mem["under"] = out, under
        return out
    if wall_blocked(frame, dest, own):
        dest = box
    else:
        d = conveyor_dir(frame, dest)
        while d:
            nxt = (dest[0] + d[0], dest[1] + d[1], dest[2], dest[3])
            if wall_blocked(frame, nxt, own):
                break
            dest = nxt
    run = burn(out, bars[0]) if bars else []
    if dest == box:
        _mem["frame"], _mem["under"] = out, under
        return out
    rings = find(state, lambda o: o["type"] == "refuel" and contains(dest, o))
    tokens = find(state, lambda o: o["type"] == "button" and contains(dest, o))
    collects = find(state, lambda o: "collect" in o.get("tags", []) and contains(dest, o))
    consumed = set()
    for o in rings + tokens:
        consumed |= set(cells((o["x"], o["y"], o["w"], o["h"])))
    new_under = {c: (FLOOR if c in consumed else frame[c[1]][c[0]]) for c in cells(dest)}
    for c in cells(box):
        out[c[1]][c[0]] = under.get(c, FLOOR) if under else FLOOR
    for j in range(p["h"]):
        for i in range(p["w"]):
            out[dest[1] + j][dest[0] + i] = p["pixels"][j][i]
    if rings and bars:
        refill(out, bars[0], run)
    g = find(state, lambda o: o["name"] == "hud_glyph")
    if g:
        m, col = glyph_mask(frame, g[0])
        for _ in collects:
            m = rotate_cw(m)
        if tokens:
            col = 9
        if collects or tokens:
            draw_glyph(out, g[0], m, col)
    _mem["frame"], _mem["under"] = out, new_under
    return out
