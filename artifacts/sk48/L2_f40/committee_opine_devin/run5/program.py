# Mechanics: crane ring (player) moves y+-6 (A1/A2) carrying blocks on its beam arm; A4/A3 extend/retract the beam by 6.
# Blocks pushed by moving rects chain; a vertical move cancels if any block would leave the colour-4 field (block A1 no_change).
# Extend pushes swept next/collected blocks (todo blocks pass under); hooked next/collected block at beam end-5 drags on-beam
# blocks, a blocked group leaves the beam to move alone. Tags: hooked next alone among non-collected -> collected; revert off beam.
# HUD row: one 2->3 from x=63 per 3 non-click actions ((n-1)//3, n hidden, continuity-gated); legend swatch centre 0 = collected.
DY = {1: -6, 2: 6}
DW = {4: 6, 3: -6}
_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def hit(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def hud_row(frame):
    for y, row in enumerate(frame):
        if all(v not in (4, 5) for v in row):
            return y
    return len(frame)


def field_of(frame, hy):
    cells = [(x, y) for y in range(hy) for x, v in enumerate(frame[y]) if v == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def inside(r, f):
    return r[0] >= f[0] and r[1] >= f[1] and r[0] + r[2] - 1 <= f[2] and r[1] + r[3] - 1 <= f[3]


def tag_of(b):
    for t in ("todo", "next", "collected"):
        if t in b["tags"]:
            return t
    return None


def on_beam(b, beam):
    return hit(rect(b), rect(beam))


def is_hooked(b, beam):
    return (tag_of(b) in ("next", "collected") and on_beam(b, beam)
            and b["x"] == beam["x"] + beam["w"] - 5)


def push_chain(blocks, movers, solids, dx, dy, f, ring):
    moved = set(movers)
    changed = True
    while changed:
        changed = False
        new = [solid for solid in solids]
        for b in blocks:
            if b["name"] in moved:
                new.append((b["x"] + dx, b["y"] + dy, b["w"], b["h"]))
        for b in blocks:
            if b["name"] not in moved and any(hit(rect(b), r) for r in new):
                moved.add(b["name"])
                changed = True
    for b in blocks:
        if b["name"] in moved:
            r = (b["x"] + dx, b["y"] + dy, b["w"], b["h"])
            if not inside(r, f) or hit(r, rect(ring)):
                return None
    return moved


def step(state, action, f):
    ring = next(o for o in state if o["type"] == "player")
    beam = next(o for o in state if o["type"] == "arm")
    blocks = [o for o in state if o["type"] == "block"]
    if action in DY:
        dy = DY[action]
        nr = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
        if nr[1] < f[1] or nr[1] + nr[3] - 1 > f[3]:
            return
        nb = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
        carried = [b["name"] for b in blocks if on_beam(b, beam)]
        moved = push_chain(blocks, carried, [nr, nb], 0, dy, f, {"x": -99, "y": -99, "w": 0, "h": 0})
        if moved is None:
            return
        ring["y"] += dy
        beam["y"] += dy
        for b in blocks:
            if b["name"] in moved:
                b["y"] += dy
    elif action in DW:
        dw = DW[action]
        nw = beam["w"] + dw
        if nw < 1 or beam["x"] + nw - 1 > f[2]:
            return
        hooked = [b for b in blocks if is_hooked(b, beam)]
        moved = None
        if hooked:
            group = [b["name"] for b in blocks if on_beam(b, beam)]
            moved = push_chain(blocks, group, [], dw, 0, f, ring)
        elif dw > 0:
            end = beam["x"] + beam["w"]
            sw = (end, beam["y"], dw, beam["h"])
            swept = [b["name"] for b in blocks
                     if tag_of(b) in ("next", "collected") and hit(rect(b), sw)]
            if swept:
                moved = push_chain(blocks, swept, [], dw, 0, f, ring)
        beam["w"] = nw
        for b in blocks:
            if moved and b["name"] in moved:
                b["x"] += dw
    else:
        return
    update_tags(blocks, beam)


def update_tags(blocks, beam, order=None):
    order = sorted(blocks, key=lambda b: b.get("_lx", b["x"]))
    hooked = [b for b in blocks if is_hooked(b, beam)]
    for h in hooked:
        others = [b for b in blocks if b is not h and on_beam(b, beam)]
        if tag_of(h) == "next" and all(tag_of(b) == "collected" for b in others):
            set_tag(h, "collected")
            for b in order:
                if tag_of(b) == "todo":
                    set_tag(b, "next")
                    break
    if not hooked:
        for b in blocks:
            if tag_of(b) == "collected" and not on_beam(b, beam):
                for o in blocks:
                    if tag_of(o) == "next":
                        set_tag(o, "todo")
                set_tag(b, "next")


def set_tag(b, t):
    b["tags"] = [t if x in ("todo", "next", "collected") else x for x in b["tags"]]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def background(x, y, f, ring):
    if f[0] <= x <= f[2] and f[1] <= y <= f[3]:
        return 4
    if ring["x"] + 2 <= x <= ring["x"] + 3 and f[1] + 2 <= y <= f[3] - 2:
        return 2 if (y - f[1] - 2) % 6 < 2 else 3
    return 5


def legend(frame, hy, colour):
    cells = [(x, y) for y in range(hy + 1, len(frame)) for x, v in enumerate(frame[y]) if v == colour]
    if not cells:
        return None
    return min(c[0] for c in cells), min(c[1] for c in cells)


def transition_function(state, action, frame):
    hy = hud_row(frame)
    f = field_of(frame, hy)
    marks = sum(1 for v in frame[hy] if v == 3)
    if _mem["frame"] is not None and _mem["frame"] == frame:
        n = _mem["n"]
    else:
        n = 3 * marks + 1 if marks else 0
    st = [dict(o, tags=list(o.get("tags", []))) for o in state]
    for b in st:
        if b["type"] == "block":
            c = b["pixels"][0][0]
            lg = legend(frame, hy, c)
            b["_c"] = c
            b["_lg"] = lg
            b["_lx"] = lg[0] if lg else b["x"]
    act = action if isinstance(action, int) else 6
    step(st, act, f)
    out = [list(r) for r in frame]
    ring0 = next(o for o in state if o["type"] == "player")
    for o in state:
        for y in range(o["y"], o["y"] + o["h"]):
            for x in range(o["x"], o["x"] + o["w"]):
                if 0 <= y < 64 and 0 <= x < 64:
                    out[y][x] = background(x, y, f, ring0)
    for o in sorted(st, key=lambda o: o.get("layer", 0)):
        px = beam_pixels(o["w"]) if o["type"] == "arm" else o["pixels"]
        for j in range(o["h"]):
            for i in range(o["w"]):
                if 0 <= o["y"] + j < 64 and 0 <= o["x"] + i < 64 and px[j][i] >= 0:
                    out[o["y"] + j][o["x"] + i] = px[j][i]
    for b in st:
        if b["type"] == "block" and b["_lg"]:
            lx, ly = b["_lg"]
            v = 0 if tag_of(b) == "collected" else b["_c"]
            for yy in (ly + 1, ly + 2):
                for xx in (lx + 1, lx + 2):
                    out[yy][xx] = v
    if act != 6:
        n += 1
    m = max(0, (n - 1) // 3)
    for k in range(m):
        if 0 <= 63 - k and out[hy][63 - k] in (2, 3):
            out[hy][63 - k] = 3
    _mem["frame"] = out
    _mem["n"] = n
    return out
