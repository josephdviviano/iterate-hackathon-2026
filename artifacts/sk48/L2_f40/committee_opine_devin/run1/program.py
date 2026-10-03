# Mechanics: crane ring (player) + beam (arm, h=2, w=1+6k) + 4x4 blocks on a colour-4 playfield read from the frame.
# A1/A2 move ring+beam by -/+6 carrying blocks under the beam and pushing blocks hit (chain); any block leaving the field cancels all.
# A4/A3 grow/shrink beam by 6; a hooked next/collected block (x == tip-5) drags all on-beam blocks, else extend pushes swept blocks;
# a blocked group stays while the beam still moves. Next hooked with no todo on beam -> collected (legend centre 0); unhooked reverts.
# HUD row (first all-2/3 row) gains a 3 from the right: (n-1)//3, n = non-click actions; n is hidden (continuity-gated, fallback 3b+1).
STEP = 6
_memo = {"frame": None, "n": 0}


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


class World:
    def __init__(self, frame):
        self.f = frame
        self.bar = next(y for y in range(64) if all(v in (2, 3) for v in frame[y]))
        cells = [(x, y) for y in range(self.bar) for x in range(64) if frame[y][x] == 4]
        self.L = min(x for x, _ in cells)
        self.R = max(x for x, _ in cells)
        self.T = min(y for _, y in cells)
        self.B = max(y for _, y in cells)

    def block_ok(self, r, ring):
        x, y, w, h = r
        return (x >= self.L and x + w <= self.R + 1 and y >= self.T and y + h <= self.B + 1
                and not overlap(r, ring))

    def ring_ok(self, r):
        return r[1] >= self.T and r[1] + r[3] <= self.B + 1

    def bg(self, x, y, ring):
        if self.L <= x <= self.R and self.T <= y <= self.B:
            return 4
        if x in (ring["x"] + 2, ring["x"] + 3) and self.T + 2 <= y <= self.B - 2:
            return 2 if (y - self.T - 2) % 6 < 2 else 3
        return 5


def push_chain(blocks, movers, extra, d, world, ring_r):
    """Move movers by d; blocks overlapping moved rects or extra rects join. Returns new rects or None."""
    dx, dy = d
    pos = {i: rect(b) for i, b in enumerate(blocks)}
    moved = set(movers)
    changed = True
    while changed:
        changed = False
        new = [(pos[i][0] + dx, pos[i][1] + dy, 4, 4) for i in moved] + list(extra)
        for i in pos:
            if i not in moved and any(overlap(pos[i], r) for r in new):
                moved.add(i)
                changed = True
    out = dict(pos)
    for i in moved:
        out[i] = (pos[i][0] + dx, pos[i][1] + dy, 4, 4)
        if not world.block_ok(out[i], ring_r):
            return None
    return out


def status(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def step(state, action, world):
    ring = dict(next(o for o in state if o["type"] == "player"))
    beam = dict(next(o for o in state if o["type"] == "arm"))
    blocks = [dict(o) for o in state if o["type"] == "block"]
    ring_r, beam_r = rect(ring), rect(beam)
    end = beam["x"] + beam["w"]

    def on_beam(b):
        return overlap(rect(b), beam_r)

    def fully_on(b):
        return on_beam(b) and b["x"] >= beam["x"] and b["x"] + 4 <= end

    hooked = any(status(b) in ("next", "collected") and b["x"] == end - 5 and on_beam(b) for b in blocks)
    if action in (1, 2):
        dy = -STEP if action == 1 else STEP
        nr = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
        nb = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
        carried = [i for i, b in enumerate(blocks) if on_beam(b)]
        out = push_chain(blocks, carried, [nr, nb], (0, dy), world, nr) if world.ring_ok(nr) else None
        if out is not None:
            ring["y"] += dy
            beam["y"] += dy
            for i, r in out.items():
                blocks[i]["x"], blocks[i]["y"] = r[0], r[1]
    elif action in (3, 4):
        dw = STEP if action == 4 else -STEP
        nw = beam["w"] + dw
        if 1 <= nw and beam["x"] + nw <= world.R + 1:
            movers = [i for i, b in enumerate(blocks) if fully_on(b)] if hooked else []
            extra = [(end, beam["y"], STEP, beam["h"])] if dw > 0 else []
            out = push_chain(blocks, movers, extra, (dw, 0), world, ring_r) if (movers or extra) else None
            beam["w"] = nw
            if out is not None:
                for i, r in out.items():
                    blocks[i]["x"], blocks[i]["y"] = r[0], r[1]
    update_tags(beam, blocks, world)
    return ring, beam, blocks


def legend_order(world, blocks):
    colours = {b["pixels"][0][0] for b in blocks}
    first = {}
    for y in range(world.bar + 1, 64):
        for x in range(64):
            c = world.f[y][x]
            if c in colours and c not in first:
                first[c] = (x, y)
    return sorted(first, key=lambda c: first[c]), first


def update_tags(beam, blocks, world):
    end = beam["x"] + beam["w"]
    br = rect(beam)
    order, _ = legend_order(world, blocks)
    by_col = {b["pixels"][0][0]: b for b in blocks}

    def hooked(b):
        return b["x"] == end - 5 and overlap(rect(b), br)

    def set_tag(b, t):
        b["tags"] = [x for x in b["tags"] if x not in ("todo", "next", "collected")] + [t]

    for b in blocks:
        if status(b) == "collected" and not hooked(b):
            for o in blocks:
                if status(o) == "next":
                    set_tag(o, "todo")
            set_tag(b, "next")
    nxt = [b for b in blocks if status(b) == "next"]
    todo_on = any(status(b) == "todo" and overlap(rect(b), br) for b in blocks)
    if nxt and hooked(nxt[0]) and not todo_on:
        set_tag(nxt[0], "collected")
        for c in order:
            if c in by_col and status(by_col[c]) == "todo":
                set_tag(by_col[c], "next")
                break


def paint(out, o):
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            x, y = o["x"] + i, o["y"] + j
            if 0 <= x < 64 and 0 <= y < 64 and v >= 0:
                out[y][x] = v


def render(state, frame, world, ring, beam, blocks, n):
    out = [list(r) for r in frame]
    for o in state:
        if o["type"] in ("player", "arm", "block"):
            for j in range(o["h"]):
                for i in range(o["w"]):
                    x, y = o["x"] + i, o["y"] + j
                    if 0 <= x < 64 and 0 <= y < 64:
                        out[y][x] = world.bg(x, y, ring)
    paint(out, ring)
    beam["pixels"] = [[2 if (i + j) % 3 == 1 else 1 for i in range(beam["w"])] for j in range(beam["h"])]
    paint(out, beam)
    for b in blocks:
        paint(out, b)
    filled = max(0, (n - 1) // 3)
    for k in range(64):
        x = 63 - k
        if out[world.bar][x] in (2, 3):
            out[world.bar][x] = 3 if k < filled else 2
    _, first = legend_order(world, blocks)
    for b in blocks:
        c = b["pixels"][0][0]
        if c in first:
            x0, y0 = first[c]
            v = 0 if status(b) == "collected" else c
            for dy in (1, 2):
                for dx in (1, 2):
                    out[y0 + dy][x0 + dx] = v
    return out


def count_bar(frame, world):
    return sum(1 for v in frame[world.bar] if v == 3)


def transition_function(state, action, frame):
    world = World(frame)
    if _memo["frame"] is not None and frame == _memo["frame"]:
        n = _memo["n"]
    else:
        b = count_bar(frame, world)
        n = 3 * b + 1 if b else 0
    act = action if isinstance(action, int) else 6
    if act != 6:
        n += 1
    ring, beam, blocks = step(state, act, world)
    out = render(state, frame, world, ring, beam, blocks, n)
    _memo["frame"], _memo["n"] = out, n
    return out
