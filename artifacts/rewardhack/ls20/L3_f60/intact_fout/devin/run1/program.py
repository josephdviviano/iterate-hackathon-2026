# Mechanics: player (5x5 sprite) moves 5 cells per A1-4 (up/down/left/right); a move is blocked when the
# target rect leaves the board, holds a wall cell (colour 4) or overlaps the legend; landing on a cell whose
# outer edge is lined with colour-1 pads slides the player away from the pads until blocked. Every action
# burns 2 cells from the left of the step_bar; refuel rings covered by the player are erased and refill the bar;
# landing on the diamond rotates the HUD glyph clockwise; the recolor token is erased and recolours the glyph.
# Unconfirmed: player inside the legend chamber freezes (no bar burn); under-player cells kept via continuity.
import copy

WALL, PAD, FLOOR = 4, 1, 3
STEP = 5
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
TAG_COLOURS = {"hud_blue": 9, "hud_red": 8, "hud_orange": 12, "hud_green": 14, "hud_yellow": 11}

_memory = {"frame": None, "under": None}


def find(state, typ, tag=None):
    return [o for o in state if o["type"] == typ and (tag is None or tag in o.get("tags", []))]


def rect_cells(x, y, w, h):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def overlaps(ax, ay, aw, ah, o):
    return ax < o["x"] + o["w"] and o["x"] < ax + aw and ay < o["y"] + o["h"] and o["y"] < ay + ah


def blocked(bg, x, y, w, h, legends):
    if x < 0 or y < 0 or x + w > 64 or y + h > 64:
        return True
    if any(bg[cy][cx] == WALL for cx, cy in rect_cells(x, y, w, h)):
        return True
    return any(overlaps(x, y, w, h, lg) for lg in legends)


def pad_push(bg, x, y, w, h):
    sides = {(1, 0): [(x - 1, y + j) for j in range(h)], (-1, 0): [(x + w, y + j) for j in range(h)],
             (0, 1): [(x + i, y - 1) for i in range(w)], (0, -1): [(x + i, y + h) for i in range(w)]}
    for d, cells in sides.items():
        if all(0 <= cx < 64 and 0 <= cy < 64 and bg[cy][cx] == PAD for cx, cy in cells):
            return d
    return None


def in_chamber(p, legends):
    for lg in legends:
        ch = {"x": lg["x"] - 3, "y": lg["y"] - 3, "w": lg["w"] + 6, "h": lg["h"] + 6}
        if overlaps(p["x"], p["y"], p["w"], p["h"], ch):
            return True
    return False


def erase_sprite(bg, o, region):
    seeds = [c for c in rect_cells(o["x"], o["y"], o["w"], o["h"]) if bg[c[1]][c[0]] != FLOOR]
    seen = set(seeds)
    while seeds:
        cx, cy = seeds.pop()
        bg[cy][cx] = FLOOR
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (cx + dx, cy + dy)
                if n in region and n not in seen and bg[n[1]][n[0]] != FLOOR:
                    seen.add(n)
                    seeds.append(n)


def glyph_cells(g):
    s = g["w"] // 3
    return s, [[(g["x"] + c * s, g["y"] + r * s) for c in range(3)] for r in range(3)]


def paint_glyph(out, g, grid):
    s, cells = glyph_cells(g)
    for r in range(3):
        for c in range(3):
            x0, y0 = cells[r][c]
            for dy in range(s):
                for dx in range(s):
                    out[y0 + dy][x0 + dx] = grid[r][c]


def read_glyph(frame, g):
    s, cells = glyph_cells(g)
    return [[frame[y][x] for x, y in row] for row in cells]


def rotate_glyph(frame, g):
    a = read_glyph(frame, g)
    paint_glyph(frame, g, [[a[2 - j][i] for j in range(3)] for i in range(3)])


def recolour_glyph(frame, g, colour):
    a = read_glyph(frame, g)
    bgc = frame[g["y"]][g["x"] - 1] if g["x"] > 0 else 5
    paint_glyph(frame, g, [[colour if v != bgc else v for v in row] for row in a])


def bar_update(frame, bar, refill):
    y0, h = bar["y"], bar["h"]
    end = bar["x"] + bar["w"] - 1
    colour = frame[y0][end] if bar["w"] > 0 else 11
    start = end
    while start - 1 >= 0 and frame[y0][start - 1] in (FLOOR, colour):
        start -= 1
    left = bar["x"]
    if refill:
        for x in range(start, end + 1):
            for y in range(y0, y0 + h):
                frame[y][x] = colour
    else:
        for x in range(left, min(left + 2, end + 1)):
            for y in range(y0, y0 + h):
                frame[y][x] = FLOOR


def transition_function(state, action, frame):
    frame = copy.deepcopy(frame)
    players = find(state, "player")
    if not players:
        return frame
    p = players[0]
    pc = rect_cells(p["x"], p["y"], p["w"], p["h"])
    if _memory["frame"] == frame and _memory["under"] is not None:
        under = dict(_memory["under"])
    else:
        under = {c: FLOOR for c in pc}
    bg = copy.deepcopy(frame)
    for (cx, cy), v in under.items():
        bg[cy][cx] = v
    legends = find(state, "target", "legend")
    bars = find(state, "counter", "budget")
    glyphs = find(state, "counter", "progress")

    if in_chamber(p, legends):
        _memory["frame"], _memory["under"] = frame, under
        return frame

    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
    moved = False
    if action in DIRS:
        dx, dy = DIRS[action]
        if not blocked(bg, x + dx * STEP, y + dy * STEP, w, h, legends):
            x, y, moved = x + dx * STEP, y + dy * STEP, True
            push = pad_push(bg, x, y, w, h)
            while push and not blocked(bg, x + push[0] * STEP, y + push[1] * STEP, w, h, legends):
                x, y = x + push[0] * STEP, y + push[1] * STEP

    out = bg
    region = set(rect_cells(x, y, w, h))
    refill = False
    if moved:
        for o in state:
            if o is p or not overlaps(x, y, w, h, o):
                continue
            tags = o.get("tags", [])
            if o["type"] == "refuel":
                erase_sprite(out, o, region)
                refill = True
            elif o["type"] == "target" and "collect" in tags:
                for g in glyphs:
                    rotate_glyph(out, g)
            elif o["type"] == "button" and "recolor" in tags:
                erase_sprite(out, o, region)
                colour = next((TAG_COLOURS[t] for t in tags if t in TAG_COLOURS), None)
                if colour is not None:
                    for g in glyphs:
                        recolour_glyph(out, g, colour)
    for bar in bars:
        bar_update(out, bar, refill)

    new_under = {c: out[c[1]][c[0]] for c in region}
    for j, row in enumerate(p["pixels"]):
        for i, v in enumerate(row):
            if v >= 0:
                out[y + j][x + i] = v
    _memory["frame"], _memory["under"] = copy.deepcopy(out), new_under
    return out
