# Mechanics: crane ring (player) moves y-+6 (A1/A2) carrying blocks on its beam (arm); A4/A3 grow/shrink the beam by 6.
# Blocks overlapping moved things are pushed (chains); a vertical move whose group leaves the playfield is cancelled
# (player A1 no_change = carried blocks would cross the field top). The next/collected block hooked at beam end-5 is dragged
# with every on-beam block; a blocked horizontal group leaves the beam sliding alone. Collected = hooked & alone on beam.
# HUD: row of 2s gains a 3 from the right, (n-1)//3, n = non-click actions (hidden, continuity-gated); unconfirmed: A5/A7.
MEM = {"frame": None, "n": 0}
STEP = 6


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def hud_row(frame):
    for y, row in enumerate(frame):
        if all(v not in (4, 5) for v in row):
            return y
    return len(frame)


def field_rect(frame, hy):
    cells = [(x, y) for y in range(hy) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def parse(state):
    ring = beam = None
    blocks = []
    for o in state:
        if o["type"] == "player":
            ring = dict(o)
        elif o["type"] == "arm":
            beam = dict(o)
        elif o["type"] == "block":
            b = dict(o)
            b["tag"] = [t for t in o["tags"] if t in ("todo", "next", "collected")][0]
            b["col"] = o["pixels"][0][0]
            blocks.append(b)
    return ring, beam, blocks


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def in_field(r, F):
    return r[0] >= F[0] and r[1] >= F[1] and r[0] + r[2] <= F[2] + 1 and r[1] + r[3] <= F[3] + 1


def push_chain(movers, seeds, blocks, dx, dy):
    """movers: rects (already at new position) that push; seeds: blocks moving by (dx,dy)."""
    moving = list(seeds)
    changed = True
    while changed:
        changed = False
        pushers = list(movers) + [rect(b, dx, dy) for b in moving]
        for b in blocks:
            if any(b is m for m in moving):
                continue
            if any(overlap(rect(b), p) for p in pushers):
                moving.append(b)
                changed = True
    return moving


def on_beam(beam, blocks):
    return [b for b in blocks if overlap(rect(b), rect(beam))]


def hooked(beam, blocks):
    end = beam["x"] + beam["w"]
    for b in on_beam(beam, blocks):
        if b["tag"] in ("next", "collected") and b["x"] == end - 5:
            return b
    return None


def step_vertical(ring, beam, blocks, dy, F):
    carried = on_beam(beam, blocks)
    nr, nb = rect(ring, 0, dy), rect(beam, 0, dy)
    moving = push_chain([nr, nb], carried, blocks, 0, dy)
    if nr[1] < F[1] or nr[1] + nr[3] > F[3] + 1:
        return False
    if not all(in_field(rect(b, 0, dy), F) for b in moving):
        return False
    ring["y"] += dy
    beam["y"] += dy
    for b in moving:
        b["y"] += dy
    return True


def group_ok(moving, ring, dx, F):
    return all(in_field(rect(b, dx, 0), F) and not overlap(rect(b, dx, 0), rect(ring)) for b in moving)


def step_horizontal(ring, beam, blocks, dw, F):
    nw = beam["w"] + dw
    if nw < 1 or beam["x"] + nw > F[2] + 1:
        return False
    hk = hooked(beam, blocks)
    end = beam["x"] + beam["w"]
    if hk is not None:
        moving = push_chain([], on_beam(beam, blocks), blocks, dw, 0)
    elif dw > 0:
        swept = (end, beam["y"], dw, beam["h"])
        moving = push_chain([swept], [], blocks, dw, 0)
    else:
        moving = []
    if moving and group_ok(moving, ring, dw, F):
        for b in moving:
            b["x"] += dw
    beam["w"] = nw
    return True


def update_tags(beam, blocks, order):
    hk = hooked(beam, blocks)
    alone = hk is not None and len(on_beam(beam, blocks)) == 1
    found = False
    for c in order:
        b = next((b for b in blocks if b["col"] == c), None)
        if b is None:
            continue
        if found:
            b["tag"] = "todo"
        elif b["tag"] in ("next", "collected"):
            if b is hk and alone:
                b["tag"] = "collected"
            else:
                b["tag"] = "next"
                found = True
        elif b["tag"] == "todo":
            b["tag"] = "next"
            found = True


def background(x, y, F, rail_cols):
    if F[0] <= x <= F[2] and F[1] <= y <= F[3]:
        return 4
    if x in rail_cols and F[1] + 2 <= y <= F[3] - 2:
        return 2 if (y - F[1] - 2) % 6 < 2 else 3
    return 5


def beam_cell(i, j):
    return 2 if (j == 0 and i % 3 == 1) or (j == 1 and i % 3 == 0) else 1


def legend(frame, hy, colours):
    pos = {}
    for c in colours:
        cells = [(x, y) for y in range(hy + 1, 64) for x in range(64) if frame[y][x] == c]
        if cells:
            pos[c] = (min(p[0] for p in cells), min(p[1] for p in cells))
    return pos


def transition_function(state, action, frame):
    out = [list(map(int, row)) for row in frame]
    hy = hud_row(frame)
    bar = sum(1 for v in frame[hy] if v == 3) if hy < 64 else 0
    if MEM["frame"] == frame:
        n = MEM["n"]
    else:
        n = 3 * bar + 1 if bar else 0
    if isinstance(action, dict):
        MEM["frame"], MEM["n"] = out, n
        return [row[:] for row in out]
    n += 1
    ring, beam, blocks = parse(state)
    F = field_rect(frame, hy)
    rail_cols = (ring["x"] + 2, ring["x"] + 3)
    pos = legend(frame, hy, [b["col"] for b in blocks])
    order = sorted(pos, key=lambda c: pos[c][0])
    old = [rect(ring), rect(beam)] + [rect(b) for b in blocks]
    if action == 1:
        step_vertical(ring, beam, blocks, -STEP, F)
    elif action == 2:
        step_vertical(ring, beam, blocks, STEP, F)
    elif action == 4:
        step_horizontal(ring, beam, blocks, STEP, F)
    elif action == 3:
        step_horizontal(ring, beam, blocks, -STEP, F)
    update_tags(beam, blocks, order)
    for (x0, y0, w, h) in old:
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                if 0 <= x < 64 and 0 <= y < hy:
                    out[y][x] = background(x, y, F, rail_cols)
    for j, prow in enumerate(ring["pixels"]):
        for i, v in enumerate(prow):
            out[ring["y"] + j][ring["x"] + i] = int(v)
    for j in range(beam["h"]):
        for i in range(beam["w"]):
            out[beam["y"] + j][beam["x"] + i] = beam_cell(i, j)
    for b in blocks:
        for j in range(b["h"]):
            for i in range(b["w"]):
                out[b["y"] + j][b["x"] + i] = b["col"]
    for b in blocks:
        if b["col"] in pos:
            lx, ly = pos[b["col"]]
            v = 0 if b["tag"] == "collected" else b["col"]
            for j in (1, 2):
                for i in (1, 2):
                    out[ly + j][lx + i] = v
    if hy < 64:
        k = max(0, (n - 1) // 3)
        for x in range(64):
            if out[hy][x] in (2, 3):
                out[hy][x] = 3 if x >= 64 - k else 2
    MEM["frame"], MEM["n"] = out, n
    return [row[:] for row in out]
