# Mechanics: ring (player) moves y+-6 on A1/A2 carrying blocks on the beam; beam (arm) w+-6 on A4/A3.
# Blocks are pushed by a closure push chain; a vertical move cancels if any block leaves the floor (colour 4
# field); a horizontal group move that fails leaves the beam sliding alone. A next/collected block at tip-4
# is hooked and drags all on-beam blocks. Tags: collect / revert -> legend swatch centre 2x2 = 0. HUD row
# gets one 3 from the right every 3 non-click actions; counter phase is hidden (continuity-gated, unconfirmed).
import copy

_mem = {"frame": None, "n": None}


def field_bounds(frame):
    sep = next(y for y in range(64) if all(v not in (4, 5) for v in frame[y]))
    cells = [(x, y) for y in range(sep) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return sep, min(xs), min(ys), max(xs), max(ys)


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def status(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def push_chain(blocks, movers, solids, dx, dy):
    moving = set(movers)
    changed = True
    while changed:
        changed = False
        hit = list(solids) + [rect(blocks[i], dx, dy) for i in moving]
        for i, b in enumerate(blocks):
            if i not in moving and any(overlap(rect(b), r) for r in hit):
                moving.add(i)
                changed = True
    return moving


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def step(ring, beam, blocks, action, fb):
    _, left, top, right, bot = fb

    def block_ok(b):
        return (b["x"] >= max(left, ring["x"] + ring["w"]) and b["x"] + b["w"] - 1 <= right
                and b["y"] >= top and b["y"] + b["h"] - 1 <= bot)

    def on_beam(b, bm):
        return overlap(rect(b), rect(bm))

    if action in (1, 2):
        dy = -6 if action == 1 else 6
        if ring["y"] + dy < top or ring["y"] + dy + ring["h"] - 1 > bot:
            return False
        carried = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        mv = push_chain(blocks, carried, [rect(ring, 0, dy), rect(beam, 0, dy)], 0, dy)
        trial = copy.deepcopy(blocks)
        for i in mv:
            trial[i]["y"] += dy
        if not all(block_ok(trial[i]) for i in mv):
            return False
        ring["y"] += dy
        beam["y"] += dy
        blocks[:] = trial
        return True
    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nw = beam["w"] + dw
        if nw < 1 or beam["x"] + nw - 1 > right:
            return False
        tip = beam["x"] + beam["w"] - 5
        hooked = any(on_beam(b, beam) and b["x"] == tip and status(b) in ("next", "collected") for b in blocks)
        if hooked:
            mv = push_chain(blocks, [i for i, b in enumerate(blocks) if on_beam(b, beam)], [], dw, 0)
        elif dw > 0:
            swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
            mv = push_chain(blocks, [], [swept], dw, 0)
        else:
            mv = set()
        trial = copy.deepcopy(blocks)
        for i in mv:
            trial[i]["x"] += dw
        if all(block_ok(trial[i]) for i in mv):
            blocks[:] = trial
        beam["w"] = nw
        beam["pixels"] = beam_pixels(nw)
        return True
    return False


def set_status(b, s):
    b["tags"] = [t for t in b["tags"] if t not in ("collected", "next", "todo")] + [s]


def update_tags(beam, blocks, order):
    onb = [b for b in blocks if overlap(rect(b), rect(beam))]
    tip = beam["x"] + beam["w"] - 5
    for b in onb:
        if status(b) == "next" and b["x"] == tip and all(status(o) == "collected" for o in onb if o is not b):
            set_status(b, "collected")
            todo = [o for o in blocks if status(o) == "todo"]
            todo.sort(key=lambda o: order.get(color(o), 99))
            if todo:
                set_status(todo[0], "next")
            break
    for b in blocks:
        if status(b) == "collected" and b not in onb:
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")


def color(b):
    return b["pixels"][0][0]


def legend(frame, sep, blocks):
    out = {}
    for b in blocks:
        c = color(b)
        cells = [(x, y) for y in range(sep + 1, 64) for x in range(64) if frame[y][x] == c]
        if cells:
            out[c] = (min(p[0] for p in cells), min(p[1] for p in cells))
    return out


def background(x, y, fb, ring):
    _, left, top, right, bot = fb
    if left <= x <= right and top <= y <= bot:
        return 4
    if x in (ring["x"] + 2, ring["x"] + 3) and top + 2 <= y <= bot - 2:
        return 2 if (y - top - 2) % 6 < 2 else 3
    return 5


def draw(out, o):
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            x, y = o["x"] + i, o["y"] + j
            if 0 <= x < 64 and 0 <= y < 64 and v >= 0:
                out[y][x] = v


def transition_function(state, action, frame):
    fb = field_bounds(frame)
    sep = fb[0]
    bar = sum(1 for v in frame[sep] if v == 3)
    if _mem["frame"] == frame and _mem["n"] is not None:
        n = _mem["n"]
    else:
        n = 3 * bar + 1 if bar else 0
    objs = copy.deepcopy(state)
    ring = next(o for o in objs if o["type"] == "player")
    beam = next(o for o in objs if o["type"] == "arm")
    blocks = [o for o in objs if o["type"] == "block"]
    old = [copy.deepcopy(o) for o in (ring, beam, *blocks)]
    leg = legend(frame, sep, blocks)
    order = {c: i for i, c in enumerate(sorted(leg, key=lambda c: leg[c][0]))}
    is_click = isinstance(action, dict)
    if not is_click:
        n += 1
        if step(ring, beam, blocks, action, fb):
            update_tags(beam, blocks, order)
    out = [list(r) for r in frame]
    for o in old:
        for j in range(o["h"]):
            for i in range(o["w"]):
                x, y = o["x"] + i, o["y"] + j
                if 0 <= x < 64 and 0 <= y < 64:
                    out[y][x] = background(x, y, fb, ring)
    for o in sorted([ring, beam] + blocks, key=lambda o: o["layer"]):
        draw(out, o)
    marks = max(0, (n - 1) // 3)
    for k in range(min(marks, 64)):
        out[sep][63 - k] = 3
    for b in blocks:
        c = color(b)
        if c in leg:
            lx, ly = leg[c]
            v = 0 if status(b) == "collected" else c
            for yy in (ly + 1, ly + 2):
                for xx in (lx + 1, lx + 2):
                    out[yy][xx] = v
    _mem["frame"] = [list(r) for r in out]
    _mem["n"] = n
    return out
