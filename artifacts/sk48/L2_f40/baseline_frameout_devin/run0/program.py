# Mechanics: crane ring (player) moves y+-6 carrying blocks overlapping the beam (arm) and pushing blocks it hits (chain);
# beam w+-6: a hooked next/collected block (x == tip-4) drags all on-beam blocks, else extension pushes swept blocks;
# blocked groups (leave board) stay and the beam changes alone; blocked vertical moves cancel. Collect = hooked next with
# every other on-beam block collected (legend centre -> 0, next = following legend colour); off-beam collected reverts.
# Frame: erase via mirror/period-6 background, redraw by layer; bar row gains a 3 every 3rd non-click action (phase hidden).
import copy

BEAM_PAT = ([1, 2, 1], [2, 1, 1])
_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def status(o):
    return o["tags"][-1]


def set_status(o, s):
    o["tags"] = o["tags"][:-1] + [s]


class Layout:
    def __init__(self, frame):
        self.bar_row = next(y for y in range(64) if all(v in (2, 3) for v in frame[y]))
        cells = [(x, y) for y in range(self.bar_row) for x in range(64) if frame[y][x] == 4]
        self.left = min(c[0] for c in cells)
        self.right = max(c[0] for c in cells)
        self.top = min(c[1] for c in cells)
        self.bot = max(c[1] for c in cells)

    def inside(self, r):
        return r[0] >= self.left and r[0] + r[2] <= self.right and r[1] >= self.top and r[1] + r[3] <= self.bot

    def ring_ok(self, r):
        return r[1] >= self.top and r[1] + r[3] - 1 <= self.bot


def legend(frame, lay, colours):
    out = []
    for c in colours:
        cs = [(x, y) for y in range(lay.bar_row + 1, 64) for x in range(64) if frame[y][x] == c]
        if cs:
            x0, x1 = min(p[0] for p in cs), max(p[0] for p in cs)
            y0, y1 = min(p[1] for p in cs), max(p[1] for p in cs)
            out.append((x0, c, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)))
    out.sort()
    return [(c, r) for _, c, r in out]


def push_chain(blocks, movers, solids, dx, dy):
    moved = set(movers)
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
    changed = True
    while changed:
        changed = False
        hit = [rect(blocks[i]) for i in moved] + solids
        for i, b in enumerate(blocks):
            if i not in moved and any(overlap(rect(b), h) for h in hit):
                b["x"] += dx
                b["y"] += dy
                moved.add(i)
                changed = True
    return moved


def tip(beam):
    return beam["x"] + beam["w"] - 1


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(b, beam):
    return status(b) in ("next", "collected") and b["x"] == tip(beam) - 4 and \
        b["y"] < beam["y"] + beam["h"] and beam["y"] < b["y"] + b["h"]


def vertical(ring, beam, blocks, dy, lay):
    r2, a2, b2 = dict(ring), dict(beam), copy.deepcopy(blocks)
    carried = [i for i, b in enumerate(b2) if on_beam(b, beam)]
    r2["y"] += dy
    a2["y"] += dy
    moved = push_chain(b2, carried, [rect(r2), rect(a2)], 0, dy)
    if not lay.ring_ok(rect(r2)) or any(not lay.inside(rect(b2[i])) for i in moved):
        return ring, beam, blocks
    return r2, a2, b2


def horizontal(ring, beam, blocks, dw, lay):
    nw = beam["w"] + dw
    if nw < 1 or beam["x"] + nw - 1 > lay.right:
        return ring, beam, blocks
    a2 = dict(beam, w=nw)
    b2 = copy.deepcopy(blocks)
    if any(hooked(b, beam) for b in blocks):
        moved = push_chain(b2, [i for i, b in enumerate(b2) if on_beam(b, beam)], [], dw, 0)
    elif dw > 0:
        swept = (tip(beam) + 1, beam["y"], dw, beam["h"])
        moved = push_chain(b2, [], [swept], dw, 0)
    else:
        moved = set()
    if any(not lay.inside(rect(b2[i])) or overlap(rect(b2[i]), rect(ring)) for i in moved):
        b2 = copy.deepcopy(blocks)
    return ring, a2, b2


def update_tags(beam, blocks, order):
    by_col = {o["tags"][1]: o for o in blocks}
    for b in blocks:
        if status(b) == "next" and hooked(b, beam) and all(
                status(o) == "collected" for o in blocks if o is not b and on_beam(o, beam)):
            set_status(b, "collected")
            for c in order:
                o = by_col.get(str(c))
                if o is not None and status(o) == "todo":
                    set_status(o, "next")
                    break
            return
    if not any(hooked(b, beam) for b in blocks):
        for b in blocks:
            if status(b) == "collected" and not on_beam(b, beam):
                for o in blocks:
                    if status(o) == "next":
                        set_status(o, "todo")
                set_status(b, "next")


def beam_pixels(w):
    return [[BEAM_PAT[r][i % 3] for i in range(w)] for r in range(2)]


def background(frame, covered, lay):
    mid = lay.top + lay.bot
    vis = lambda x, y: 0 <= y < lay.bar_row and (x, y) not in covered

    def bg(x, y):
        if vis(x, mid - y):
            return frame[mid - y][x]
        for k in range(1, 11):
            for yy in (y - 6 * k, y + 6 * k):
                if vis(x, yy):
                    return frame[yy][x]
        for k in range(1, 64):
            for xx in (x - k, x + k):
                if 0 <= xx < 64 and vis(xx, y):
                    return frame[y][xx]
        return frame[y][x]
    return bg


def cells(o):
    for j in range(o["h"]):
        for i in range(o["w"]):
            yield o["x"] + i, o["y"] + j, o["pixels"][j][i]


def transition_function(state, action, frame):
    lay = Layout(frame)
    ring = next(o for o in state if o["type"] == "player")
    beam = next(o for o in state if o["type"] == "arm")
    blocks = [copy.deepcopy(o) for o in state if o["type"] == "block"]
    order = [c for c, _ in legend(frame, lay, [b["pixels"][0][0] for b in blocks])]
    bar = sum(1 for v in frame[lay.bar_row] if v == 3)
    n = _mem["n"] if _mem["frame"] == frame else (0 if bar == 0 else 3 * bar + 1)
    r2, a2, b2 = ring, beam, blocks
    if action in (1, 2):
        r2, a2, b2 = vertical(ring, beam, blocks, -6 if action == 1 else 6, lay)
    elif action in (3, 4):
        r2, a2, b2 = horizontal(ring, beam, blocks, -6 if action == 3 else 6, lay)
    if not isinstance(action, dict):
        n += 1
    update_tags(a2, b2, order)
    a2 = dict(a2, pixels=beam_pixels(a2["w"]))
    out = [list(row) for row in frame]
    covered = set((x, y) for o in state for x, y, _ in cells(o))
    bg = background(frame, covered, lay)
    for x, y in covered:
        out[y][x] = bg(x, y)
    for o in sorted([r2, a2] + b2, key=lambda o: o["layer"]):
        for x, y, v in cells(o):
            if 0 <= x < 64 and 0 <= y < 64 and v >= 0:
                out[y][x] = v
    target = max(0, (n - 1) // 3)
    row = out[lay.bar_row]
    for i in range(64):
        row[63 - i] = 3 if i < target else 2
    tags = {b["pixels"][0][0]: status(b) for b in b2}
    for c, (x0, y0, w, h) in legend(frame, lay, list(tags)):
        for y in range((h - 1) // 2, h // 2 + 1):
            for x in range((w - 1) // 2, w // 2 + 1):
                out[y0 + y][x0 + x] = 0 if tags[c] == "collected" else c
    _mem["frame"], _mem["n"] = out, n
    return out
