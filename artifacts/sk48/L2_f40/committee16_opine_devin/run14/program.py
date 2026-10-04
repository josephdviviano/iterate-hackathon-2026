# Mechanics: ring (player) + beam (arm, x=ring.x+5, y=ring.y+2, w=1+6k) + 4x4 blocks, all stateless physics.
# A1/A2 move ring+beam -/+6 carrying on-beam blocks and pushing hit blocks; any block leaving the play rect cancels.
# A4/A3 beam w +/-6 in [1, play_right-beam.x+1]; hooked next/collected block (x==tip-4) drags on-beam blocks,
# else extension pushes next/collected blocks in swept cols; blocked group -> beam changes alone. Tags: collect/revert.
# HUD: every 3rd non-click action after the 1st fills row-53 bar from the right (hidden count, continuity-gated); collected icon centre = 0.
import copy

_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def kind(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def set_kind(b, k):
    b["tags"] = [t for t in b["tags"] if t not in ("collected", "next", "todo")] + [k]


def hud_row(frame):
    for y in range(len(frame)):
        if all(v not in (4, 5) for v in frame[y]):
            return y
    return len(frame)


def play_rect(frame, sep):
    cells = [(x, y) for y in range(sep) for x in range(len(frame[0])) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)


def inside(r, play):
    return r[0] >= play[0] and r[1] >= play[1] and r[0] + r[2] <= play[0] + play[2] and r[1] + r[3] <= play[1] + play[3]


def push_chain(blocks, movers, solids, dx, dy):
    moved = set()
    frontier = []
    for b in movers:
        if id(b) not in moved:
            moved.add(id(b))
            b["x"] += dx
            b["y"] += dy
            frontier.append(b)
    pushers = list(solids) + [rect(b) for b in frontier]
    while pushers:
        nxt = []
        for b in blocks:
            if id(b) in moved:
                continue
            if any(overlap(rect(b), p) for p in pushers):
                moved.add(id(b))
                b["x"] += dx
                b["y"] += dy
                nxt.append(rect(b))
        pushers = nxt
    return [b for b in blocks if id(b) in moved]


def beam_rect(beam):
    return (beam["x"], beam["y"], beam["w"], beam["h"])


def on_beam(b, beam):
    return overlap(rect(b), beam_rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    return [b for b in blocks if kind(b) in ("next", "collected") and b["x"] == tip - 4 and on_beam(b, beam)]


def update_tags(blocks, beam):
    hk = hooked(blocks, beam)
    nxt = [b for b in blocks if kind(b) == "next"]
    if hk:
        for b in hk:
            if kind(b) == "next" and all(kind(o) == "collected" for o in blocks if o is not b and on_beam(o, beam)):
                set_kind(b, "collected")
                todo = [o for o in blocks if kind(o) == "todo"]
                if todo:
                    set_kind(max(todo, key=lambda o: o["x"]), "next")
        return
    off = [b for b in blocks if kind(b) == "collected" and not on_beam(b, beam)]
    if off:
        for b in nxt:
            set_kind(b, "todo")
        set_kind(max(off, key=lambda o: o["x"]), "next")
        return
    if not nxt:
        rest = [o for o in blocks if kind(o) == "todo"]
        if rest:
            set_kind(max(rest, key=lambda o: o["x"]), "next")


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def step(ring, beam, blocks, action, play):
    if action in (1, 2):
        dy = -6 if action == 1 else 6
        r2, b2, bl2 = dict(ring), dict(beam), copy.deepcopy(blocks)
        carried = [b for b in bl2 if on_beam(b, beam)]
        r2["y"] += dy
        b2["y"] += dy
        moved = push_chain(bl2, carried, [rect(r2), beam_rect(b2)], 0, dy)
        if not (play[1] <= r2["y"] and r2["y"] + r2["h"] <= play[1] + play[3]) or any(not inside(rect(b), play) for b in moved):
            return ring, beam, blocks
        update_tags(bl2, b2)
        return r2, b2, bl2
    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nw = beam["w"] + dw
        if nw < 1 or beam["x"] + nw > play[0] + play[2]:
            return ring, beam, blocks
        b2 = dict(beam)
        b2["w"] = nw
        b2["pixels"] = beam_pixels(nw)
        bl2 = copy.deepcopy(blocks)
        tip = beam["x"] + beam["w"] - 1
        if hooked(bl2, beam):
            moved = push_chain(bl2, [b for b in bl2 if on_beam(b, beam)], [], dw, 0)
        elif dw > 0:
            swept = (tip + 1, beam["y"], dw, beam["h"])
            hit = [b for b in bl2 if kind(b) in ("next", "collected") and overlap(rect(b), swept)]
            moved = push_chain(bl2, hit, [], dw, 0)
        else:
            moved = []
        if any(not inside(rect(b), play) or overlap(rect(b), rect(ring)) for b in moved):
            bl2 = copy.deepcopy(blocks)
        update_tags(bl2, b2)
        return ring, b2, bl2
    return ring, beam, blocks


def background(x, y, ring, play):
    if inside((x, y, 1, 1), play):
        return 4
    top, bot = play[1] + 2, play[1] + play[3] - 3
    if x in (ring["x"] + 2, ring["x"] + 3) and top <= y <= bot:
        return 2 if (y - top) % 6 < 2 else 3
    return 5


def paint(frame, o):
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            x, y = o["x"] + i, o["y"] + j
            if v >= 0 and 0 <= y < len(frame) and 0 <= x < len(frame[0]):
                frame[y][x] = v


def transition_function(state, action, frame):
    st = copy.deepcopy(state)
    ring = next(o for o in st if o["type"] == "player")
    beam = next(o for o in st if o["type"] == "arm")
    blocks = [o for o in st if o["type"] == "block"]
    sep = hud_row(frame)
    play = play_rect(frame, sep)
    out = [row[:] for row in frame]
    bar = sum(1 for v in frame[sep] if v == 3) if sep < len(frame) else 0
    n = _mem["n"] if _mem["frame"] == frame else (3 * bar + 1 if bar else 0)
    is_click = isinstance(action, dict)
    if not is_click:
        r2, b2, bl2 = step(ring, beam, blocks, action, play)
        for o in [ring, beam] + blocks:
            for j in range(o["h"]):
                for i in range(o["w"]):
                    x, y = o["x"] + i, o["y"] + j
                    out[y][x] = background(x, y, ring, play)
        for o in [r2, b2] + sorted(bl2, key=lambda o: o["layer"]):
            paint(out, o)
        n += 1
        filled = max(0, (n - 1) // 3)
        if sep < len(frame):
            W = len(frame[0])
            for k in range(filled):
                if W - 1 - k >= 0:
                    out[sep][W - 1 - k] = 3
        for b in bl2:
            col = b["pixels"][0][0]
            cells = [(x, y) for y in range(sep + 1, len(frame)) for x in range(len(frame[0])) if frame[y][x] == col]
            if not cells:
                continue
            x0, x1 = min(c[0] for c in cells), max(c[0] for c in cells)
            y0, y1 = min(c[1] for c in cells), max(c[1] for c in cells)
            fill = 0 if kind(b) == "collected" else col
            for y in range(y0 + 1, y1):
                for x in range(x0 + 1, x1):
                    out[y][x] = fill
    _mem["frame"] = out
    _mem["n"] = n
    return out
