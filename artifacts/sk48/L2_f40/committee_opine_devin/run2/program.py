# Mechanics: ring (player) + beam (arm) crane; A1/A2 move ring+beam -6/+6 carrying on-beam blocks and pushing hit blocks
# (whole move cancelled if any block or the ring leaves the colour-4 play rect); A4/A3 beam w +-6 (1..43, within field);
# a hooked next/collected block (x == beam end-5) drags all on-beam blocks, else extension pushes swept blocks; a blocked
# group leaves the beam changing alone. Tags: hooked next with other on-beam blocks collected -> collected; collected off beam
# reverts. HUD row: one 2->3 from the right per 3 non-click actions (hidden count, continuity-gated). Clicks are no-ops.
import copy

_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def hud_row(frame):
    for y in range(64):
        if all(c not in (4, 5) for c in frame[y]):
            return y
    return 64


def play_rect(frame, hy):
    cells = [(x, y) for y in range(hy) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def block_ok(b, pr):
    return pr[0] <= b["x"] and b["x"] + b["w"] - 1 <= pr[2] and pr[1] <= b["y"] and b["y"] + b["h"] - 1 <= pr[3]


def on_beam(b, beam):
    return (b["y"] <= beam["y"] and b["y"] + b["h"] >= beam["y"] + beam["h"]
            and b["x"] >= beam["x"] and b["x"] + b["w"] <= beam["x"] + beam["w"])


def tag(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return None


def set_tag(b, t):
    b["tags"] = [x for x in b["tags"] if x not in ("collected", "next", "todo")] + [t]


def push_chain(blocks, movers, dx, dy, solids):
    moved = set()
    for i in movers:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
        moved.add(i)
    frontier = list(movers)
    hitters = list(solids)
    changed = True
    while changed:
        changed = False
        rs = hitters + [rect(blocks[i]) for i in moved]
        for j, b in enumerate(blocks):
            if j in moved:
                continue
            if any(overlap(rect(b), r) for r in rs):
                b["x"] += dx
                b["y"] += dy
                moved.add(j)
                changed = True
                break
    return moved


def hooked(blocks, beam):
    end = beam["x"] + beam["w"]
    for b in blocks:
        if tag(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == end - 5:
            return b
    return None


def step(ring, beam, blocks, action, pr):
    if action in (1, 2):
        dy = -6 if action == 1 else 6
        nr, nb, nbl = dict(ring), dict(beam), copy.deepcopy(blocks)
        nr["y"] += dy
        nb["y"] += dy
        if nr["y"] < pr[1] or nr["y"] + nr["h"] - 1 > pr[3]:
            return ring, beam, blocks
        carried = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        moved = push_chain(nbl, carried, 0, dy, [rect(nr), rect(nb)])
        if any(not block_ok(nbl[i], pr) for i in moved):
            return ring, beam, blocks
        return nr, nb, nbl
    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nb = dict(beam)
        nb["w"] += dw
        if nb["w"] < 1 or nb["w"] > 43 or nb["x"] + nb["w"] - 1 > pr[2]:
            return ring, beam, blocks
        nbl = copy.deepcopy(blocks)
        h = hooked(blocks, beam)
        moved = set()
        if h is not None:
            group = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
            moved = push_chain(nbl, group, dw, 0, [])
        elif dw > 0:
            end = beam["x"] + beam["w"]
            swept = (end, beam["y"], dw, beam["h"])
            hit = [i for i, b in enumerate(blocks) if overlap(rect(b), swept)]
            moved = push_chain(nbl, hit, dw, 0, [])
        rr = rect(ring)
        if any(not block_ok(nbl[i], pr) or overlap(rect(nbl[i]), rr) for i in moved):
            nbl = copy.deepcopy(blocks)
        return ring, nb, nbl
    return ring, beam, blocks


def update_tags(blocks, beam, order):
    h = hooked(blocks, beam)
    if h is not None and tag(h) == "next":
        others = [b for b in blocks if b is not h and on_beam(b, beam)]
        if all(tag(b) == "collected" for b in others):
            set_tag(h, "collected")
            nxt = [b for b in order if tag(b) == "todo"]
            if nxt:
                set_tag(nxt[0], "next")
    if h is None:
        for b in blocks:
            if tag(b) == "collected" and not on_beam(b, beam):
                for o in blocks:
                    if tag(o) == "next":
                        set_tag(o, "todo")
                set_tag(b, "next")


def background(x, y, pr, ring):
    if pr[0] <= x <= pr[2] and pr[1] <= y <= pr[3]:
        return 4
    if x in (ring["x"] + 2, ring["x"] + 3) and pr[1] + 2 <= y <= pr[3] - 2:
        return 2 if (y - pr[1] - 2) % 6 < 2 else 3
    return 5


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def legend_boxes(frame, hy, colours):
    boxes = {}
    for c in colours:
        cells = [(x, y) for y in range(hy + 1, 64) for x in range(64) if frame[y][x] == c]
        if cells:
            boxes[c] = (min(p[0] for p in cells), min(p[1] for p in cells),
                        max(p[0] for p in cells), max(p[1] for p in cells))
    return boxes


def colour_of(b):
    return b["pixels"][0][0] if b.get("pixels") else int(b["name"].split("_")[-1])


def transition_function(state, action, frame):
    ring = next(o for o in state if o["type"] == "player")
    beam = next(o for o in state if o["type"] == "arm")
    blocks = [o for o in state if o["type"] == "block"]
    hy = hud_row(frame)
    pr = play_rect(frame, hy)
    colours = [colour_of(b) for b in blocks]
    boxes = legend_boxes(frame, hy, colours)
    is_click = isinstance(action, dict)
    out = [row[:] for row in frame]
    if not is_click:
        nr, nb, nbl = step(ring, beam, blocks, action, pr)
        order = sorted(nbl, key=lambda b: boxes.get(colour_of(b), (99,))[0])
        update_tags(nbl, nb, order)
        for o in [ring, beam] + blocks:
            for yy in range(o["y"], o["y"] + o["h"]):
                for xx in range(o["x"], o["x"] + o["w"]):
                    if 0 <= xx < 64 and 0 <= yy < hy:
                        out[yy][xx] = background(xx, yy, pr, ring)
        draws = [(nr, nr["pixels"]), (nb, beam_pixels(nb["w"]))]
        draws += [(b, [[colour_of(b)] * b["w"] for _ in range(b["h"])]) for b in nbl]
        for o, pix in draws:
            for dy, row in enumerate(pix):
                for dx, c in enumerate(row):
                    xx, yy = o["x"] + dx, o["y"] + dy
                    if 0 <= xx < 64 and 0 <= yy < hy and c >= 0:
                        out[yy][xx] = c
        for b in nbl:
            c = colour_of(b)
            if c in boxes:
                x0, y0, x1, y1 = boxes[c]
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                v = 0 if tag(b) == "collected" else c
                for yy in (cy, cy + 1):
                    for xx in (cx, cx + 1):
                        out[yy][xx] = v
    # HUD bar
    bar = frame[hy] if hy < 64 else []
    filled = sum(1 for c in bar if c == 3)
    if _mem["frame"] == frame:
        n = _mem["n"]
    else:
        n = 3 * filled + 1 if filled else 0
    if not is_click:
        n += 1
    want = max(0, (n - 1) // 3)
    if hy < 64:
        k = 0
        for xx in range(63, -1, -1):
            if k >= want:
                break
            if out[hy][xx] in (2, 3):
                out[hy][xx] = 3
                k += 1
    _mem["frame"] = [row[:] for row in out]
    _mem["n"] = n
    return out
