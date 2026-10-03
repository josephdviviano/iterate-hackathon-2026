# Crane/skewer: A1/A2 move ring+beam -/+6 rows carrying on-beam blocks (push chain; cancel if a block/ring leaves the field).
# A4/A3 beam w +/-6 (1..field edge); a next/collected block at tip-4 hooks and drags on-beam blocks, else extend pushes
# swept next/collected blocks; a blocked block group lets the beam change alone. Tags: hooked next with all other on-beam
# collected -> collected; no hook & collected off beam -> reverts. Legend centre 2x2 = 0 while collected. Clicks no-op.
# HUD bar row: rightmost 2->3 per 3 non-click actions, (n-1)//3; phase n is hidden (continuity-gated, fallback 3*bars+1).
import copy

MEMO = {"frame": None, "n": 0}
FIELD, RAIL_A, RAIL_B, EMPTY = 4, 2, 3, 5


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def bar_row(frame):
    for y, row in enumerate(frame):
        if all(v in (RAIL_A, RAIL_B) for v in row):
            return y
    return len(frame)


def field_rect(frame, by):
    cells = [(x, y) for y in range(by) for x in range(len(frame[0])) if frame[y][x] == FIELD]
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
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
            b["tags"] = list(o["tags"])
            blocks.append(b)
    return ring, beam, blocks


def tag(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def set_tag(b, t):
    b["tags"] = [x for x in b["tags"] if x not in ("collected", "next", "todo")] + [t]


def colour(b):
    return b["pixels"][0][0]


def block_ok(b, fr):
    l, t, r, bt = fr
    return b["x"] >= l and b["y"] >= t and b["x"] + b["w"] - 1 <= r and b["y"] + b["h"] - 1 <= bt


def push_chain(blocks, movers, solids, dx, dy):
    """Shift movers by (dx,dy); any other block hit by a moved block or a solid rect is pushed too."""
    moved = set(movers)
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
    changed = True
    while changed:
        changed = False
        hitters = list(solids) + [rect(blocks[i]) for i in moved]
        for j, b in enumerate(blocks):
            if j in moved:
                continue
            if any(overlap(rect(b), h) for h in hitters):
                b["x"] += dx
                b["y"] += dy
                moved.add(j)
                changed = True
                break
    return moved


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    for i, b in enumerate(blocks):
        if tag(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == tip - 4:
            return i
    return None


def step(ring, beam, blocks, action, fr):
    l, t, r, bt = fr
    if action in (1, 2):
        dy = -6 if action == 1 else 6
        nb = copy.deepcopy(blocks)
        carried = [i for i, b in enumerate(nb) if on_beam(b, beam)]
        nring, nbeam = dict(ring), dict(beam)
        nring["y"] += dy
        nbeam["y"] += dy
        moved = push_chain(nb, carried, [rect(nring), rect(nbeam)], 0, dy)
        if nring["y"] < t or nring["y"] + nring["h"] - 1 > bt:
            return ring, beam, blocks
        if any(not block_ok(nb[i], fr) for i in moved):
            return ring, beam, blocks
        return nring, nbeam, nb
    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nbeam = dict(beam)
        nbeam["w"] += dw
        if nbeam["w"] < 1 or nbeam["x"] + nbeam["w"] - 1 > r:
            return ring, beam, blocks
        nb = copy.deepcopy(blocks)
        h = hooked(blocks, beam)
        if h is not None:
            group = [i for i, b in enumerate(nb) if on_beam(b, beam)]
            moved = push_chain(nb, group, [], dw, 0)
        elif dw > 0:
            swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
            hit = [i for i, b in enumerate(nb) if tag(b) in ("next", "collected") and overlap(rect(b), swept)]
            moved = push_chain(nb, hit, [], dw, 0)
        else:
            moved = set()
        if any(not block_ok(nb[i], fr) or overlap(rect(nb[i]), rect(ring)) for i in moved):
            nb = blocks
        return ring, nbeam, nb
    return ring, beam, blocks


def update_tags(blocks, beam, legend_order):
    h = hooked(blocks, beam)
    if h is not None and tag(blocks[h]) == "next":
        if all(tag(b) == "collected" for j, b in enumerate(blocks) if j != h and on_beam(b, beam)):
            set_tag(blocks[h], "collected")
            for c in legend_order:
                cand = [b for b in blocks if colour(b) == c and tag(b) == "todo"]
                if cand:
                    set_tag(cand[0], "next")
                    break
    elif h is None:
        off = [b for b in blocks if tag(b) == "collected" and not on_beam(b, beam)]
        if off:
            rank = {c: k for k, c in enumerate(legend_order)}
            last = max(off, key=lambda b: rank.get(colour(b), -1))
            for b in blocks:
                if tag(b) == "next":
                    set_tag(b, "todo")
            set_tag(last, "next")


def legend_boxes(frame, by, colours):
    boxes = {}
    for c in colours:
        cells = [(x, y) for y in range(by + 1, len(frame)) for x in range(len(frame[0])) if frame[y][x] == c]
        if cells:
            boxes[c] = (min(p[0] for p in cells), min(p[1] for p in cells))
    return boxes


def background(x, y, fr, ring):
    l, t, r, bt = fr
    if l <= x <= r and t <= y <= bt:
        return FIELD
    if x in (ring["x"] + 2, ring["x"] + 3) and t + 2 <= y <= bt - 2:
        return RAIL_A if (y - t - 2) % 6 < 2 else RAIL_B
    return EMPTY


def cells(o):
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            yield o["x"] + i, o["y"] + j, v


def beam_pixels(beam):
    return [[2 if i % 3 == (1 - j) % 3 else 1 for i in range(beam["w"])] for j in range(beam["h"])]


def render(frame, before, after, fr):
    out = [list(row) for row in frame]
    H, W = len(out), len(out[0])
    ring0 = before[0]
    for o in [before[0], before[1]] + before[2]:
        for x, y, v in cells(o):
            if 0 <= x < W and 0 <= y < H:
                out[y][x] = background(x, y, fr, ring0)
    ring, beam, blocks = after
    beam = dict(beam)
    beam["pixels"] = beam_pixels(beam)
    for o in sorted([ring, beam] + blocks, key=lambda o: o.get("layer", 0)):
        for x, y, v in cells(o):
            if v >= 0 and 0 <= x < W and 0 <= y < H:
                out[y][x] = v
    return out


def transition_function(state, action, frame):
    by = bar_row(frame)
    fr = field_rect(frame, by)
    ring, beam, blocks = parse(state)
    is_click = isinstance(action, dict)
    if MEMO["frame"] is not None and MEMO["frame"] == frame:
        n = MEMO["n"]
    else:
        bars = sum(1 for v in frame[by] if v == RAIL_B) if by < len(frame) else 0
        n = 3 * bars + 1 if bars else 0
    colours = sorted({colour(b) for b in blocks})
    boxes = legend_boxes(frame, by, colours)
    order = sorted(boxes, key=lambda c: boxes[c][0])
    if is_click:
        nring, nbeam, nblocks = ring, beam, copy.deepcopy(blocks)
    else:
        nring, nbeam, nblocks = step(ring, beam, blocks, action, fr)
        nblocks = copy.deepcopy(nblocks)
        update_tags(nblocks, nbeam, order)
        n += 1
    out = render(frame, (ring, beam, blocks), (nring, nbeam, nblocks), fr)
    if by < len(out):
        W = len(out[0])
        k = max(0, (n - 1) // 3)
        for x in range(W):
            if out[by][x] in (RAIL_A, RAIL_B):
                out[by][x] = RAIL_B if x >= W - k else RAIL_A
    collected = {colour(b) for b in nblocks if tag(b) == "collected"}
    for c, (bx, bY) in boxes.items():
        v = 0 if c in collected else c
        for yy in (bY + 1, bY + 2):
            for xx in (bx + 1, bx + 2):
                out[yy][xx] = v
    MEMO["frame"], MEMO["n"] = out, n
    return out
