# Mechanics: frame-level rule. Floor = colour 5; any other colour (wall 15, hazard 8, markers, blocks) blocks motion.
# Block mode (cyan 10 blocks): A1/A2 move both blocks up/down 4; A3/A4 mirror (left -4/+4, right +4/-4), each independently, dest 4x4 must be floor.
# Click inactive marker (9): blocks -> armed players (1), marker -> active (11); click another marker switches; click armed player disarms;
# Armed mode: A1-4 move the active marker 4 cells inside its 4x4 cell (dest cell must be floor). Clicks on active marker / block / empty and A5 are no-ops.
# HUD: n = actions since level start (hidden, continuity-gated); bar width 3(n+1)//7 drawn 0 at row 0 from right and row 63 from left. Unconfirmed: win/timeout.
FLOOR = 5
BLOCK, ARMED, INACTIVE, ACTIVE, BAR = 10, 1, 9, 11, 0
_mem = {"frame": None, "n": 0}


def bar_width(n):
    return 3 * (n + 1) // 7


def read_bar(frame):
    w = 0
    while w < 64 and frame[63][w] == BAR:
        w += 1
    return w


def fallback_n(w):
    if w == 0:
        return 0
    n = 0
    while bar_width(n) < w:
        n += 1
    return n


def rect_cells(x, y, w, h):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def inside(x, y):
    return 0 <= x < 64 and 0 <= y < 64


def free(frame, cells, own):
    return all(inside(x, y) and ((x, y) in own or frame[y][x] == FLOOR) for x, y in cells)


def paint(frame, cells, c):
    for x, y in cells:
        frame[y][x] = c


def hit(obj, x, y):
    return obj["x"] <= x < obj["x"] + obj["w"] and obj["y"] <= y < obj["y"] + obj["h"]


def move_blocks(frame, blocks, action):
    dy = {1: -4, 2: 4}.get(action, 0)
    dxl = {3: -4, 4: 4}.get(action, 0)
    plan = []
    for b in blocks:
        dx = dxl if b["name"] == "block_left" else -dxl
        plan.append((b, dx, dy))
    for b, dx, dy in plan:
        if dx == 0 and dy == 0:
            continue
        own = set(rect_cells(b["x"], b["y"], b["w"], b["h"]))
        dest = rect_cells(b["x"] + dx, b["y"] + dy, b["w"], b["h"])
        if free(frame, dest, own):
            paint(frame, own, FLOOR)
            paint(frame, dest, BLOCK)


def move_marker(frame, m, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action, (0, 0))
    if dx == 0 and dy == 0:
        return
    own = set(rect_cells(m["x"], m["y"], m["w"], m["h"]))
    cell = rect_cells(m["x"] - 1 + dx, m["y"] - 1 + dy, 4, 4)
    if free(frame, cell, own):
        paint(frame, own, FLOOR)
        paint(frame, rect_cells(m["x"] + dx, m["y"] + dy, m["w"], m["h"]), ACTIVE)


def click(frame, state, x, y):
    players = [o for o in state if o["type"] == "player"]
    markers = [o for o in state if o["type"] == "marker"]
    armed = any("armed" in o["tags"] for o in players)
    for m in markers:
        if hit(m, x, y):
            if "active" in m["tags"]:
                return
            for o in markers:
                if "active" in o["tags"]:
                    paint(frame, rect_cells(o["x"], o["y"], o["w"], o["h"]), INACTIVE)
            paint(frame, rect_cells(m["x"], m["y"], m["w"], m["h"]), ACTIVE)
            if not armed:
                for p in players:
                    paint(frame, rect_cells(p["x"], p["y"], p["w"], p["h"]), ARMED)
            return
    if armed and any(hit(p, x, y) for p in players):
        for p in players:
            paint(frame, rect_cells(p["x"], p["y"], p["w"], p["h"]), BLOCK)
        for o in markers:
            if "active" in o["tags"]:
                paint(frame, rect_cells(o["x"], o["y"], o["w"], o["h"]), INACTIVE)


def draw_bar(frame, w):
    for i in range(64):
        frame[0][63 - i] = BAR if i < w else FLOOR
        frame[63][i] = BAR if i < w else FLOOR


def transition_function(state, action, frame):
    before = [row[:] for row in frame]
    n = _mem["n"] if _mem["frame"] == before else fallback_n(read_bar(before))
    out = [row[:] for row in frame]
    if isinstance(action, dict):
        click(out, state, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        players = [o for o in state if o["type"] == "player"]
        if any("armed" in p["tags"] for p in players):
            for m in state:
                if m["type"] == "marker" and "active" in m["tags"]:
                    move_marker(out, m, action)
        else:
            move_blocks(out, players, action)
    n += 1
    draw_bar(out, bar_width(n))
    _mem["frame"] = [row[:] for row in out]
    _mem["n"] = n
    return out
