# Mechanics: crane ring (player) moves y-+6 with its beam (arm); beam extends/retracts by 6 (w 1..43).
# Vertical moves carry blocks on the beam and push hit blocks (chain); any block leaving the colour-4 field cancels.
# Hooked next/collected block at beam tip-5 drags on-beam blocks; else extend pushes swept next/collected blocks;
# a blocked group lets the beam change alone. Tags: collect = hooked next with all other on-beam collected; revert off-beam.
# HUD: bar row gains a 3 from the right every 3 non-click actions ((n-1)//3, n hidden, continuity-gated); legend centre 0 = collected.
import copy

MEM = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def hud_row(frame):
    for y in range(64):
        if all(v not in (4, 5) for v in frame[y]):
            return y
    return 64


def field_rect(frame, hy):
    cells = [(x, y) for y in range(hy) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def block_ok(b, fr):
    l, t, r, bt = fr
    return b["x"] >= l and b["y"] >= t and b["x"] + b["w"] - 1 <= r and b["y"] + b["h"] - 1 <= bt


def push_chain(blocks, movers, solids, dx, dy):
    moving = set(movers)
    changed = True
    while changed:
        changed = False
        hits = list(solids) + [moved_rect(blocks[i], dx, dy) for i in moving]
        for j, b in enumerate(blocks):
            if j in moving:
                continue
            if any(overlap(rect(b), h) for h in hits):
                moving.add(j)
                changed = True
    return moving


def moved_rect(b, dx, dy):
    return (b["x"] + dx, b["y"] + dy, b["w"], b["h"])


def tagof(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def settag(b, t):
    b["tags"] = [x for x in b["tags"] if x not in ("collected", "next", "todo")] + [t]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def hooked_index(blocks, beam):
    tip = beam["x"] + beam["w"]
    for i, b in enumerate(blocks):
        if tagof(b) in ("next", "collected") and b["x"] == tip - 5 and overlap(rect(b), rect(beam)):
            return i
    return None


def step(ring, beam, blocks, action, fr):
    l, t, r, bt = fr
    if action in (1, 2):
        dy = -6 if action == 1 else 6
        nring = dict(ring, y=ring["y"] + dy)
        nbeam = dict(beam, y=beam["y"] + dy)
        if nring["y"] < t or nring["y"] + nring["h"] - 1 > bt:
            return ring, beam, blocks
        carried = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
        moving = push_chain(blocks, carried, [rect(nring), rect(nbeam)], 0, dy)
        nblocks = copy.deepcopy(blocks)
        for i in moving:
            nblocks[i]["y"] += dy
        if not all(block_ok(nblocks[i], fr) for i in moving):
            return ring, beam, blocks
        return nring, nbeam, nblocks
    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nw = beam["w"] + dw
        if nw < 1 or nw > 43 or beam["x"] + nw - 1 > r:
            return ring, beam, blocks
        nbeam = dict(beam, w=nw, pixels=beam_pixels(nw))
        h = hooked_index(blocks, beam)
        if h is not None:
            movers = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
            moving = push_chain(blocks, movers, [], dw, 0)
        elif dw > 0:
            swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
            movers = [i for i, b in enumerate(blocks)
                      if tagof(b) in ("next", "collected") and overlap(rect(b), swept)]
            moving = push_chain(blocks, movers, [], dw, 0)
        else:
            moving = set()
        nblocks = copy.deepcopy(blocks)
        for i in moving:
            nblocks[i]["x"] += dw
        if not all(block_ok(nblocks[i], fr) for i in moving):
            nblocks = copy.deepcopy(blocks)
        return ring, nbeam, nblocks
    return ring, beam, blocks


def update_tags(blocks, beam, order):
    h = hooked_index(blocks, beam)
    on = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
    if h is None:
        for b in blocks:
            if tagof(b) == "collected" and not overlap(rect(b), rect(beam)):
                for c in blocks:
                    if tagof(c) == "next":
                        settag(c, "todo")
                settag(b, "next")
                return
    if h is not None and tagof(blocks[h]) == "next":
        if all(tagof(blocks[i]) == "collected" for i in on if i != h):
            settag(blocks[h], "collected")
            todo = [b for b in blocks if tagof(b) == "todo"]
            todo.sort(key=lambda b: order.get(colour(b), 99))
            if todo:
                settag(todo[0], "next")


def colour(b):
    for row in b["pixels"]:
        for v in row:
            if v >= 0:
                return v
    return -1


def legend_boxes(frame, hy):
    boxes = {}
    for y in range(hy + 1, 64):
        for x in range(64):
            v = frame[y][x]
            boxes.setdefault(v, []).append((x, y))
    return boxes


def background(frame, fr, hy, covered):
    l, t, r, bt = fr
    rails = set()
    for x in range(64):
        if l <= x <= r:
            continue
        if any(frame[y][x] in (2, 3) and (x, y) not in covered for y in range(hy)):
            rails.add(x)

    def bg(x, y):
        if l <= x <= r and t <= y <= bt:
            return 4
        if x in rails and t + 2 <= y <= bt - 2:
            return 2 if (y - t - 2) % 6 < 2 else 3
        return 5
    return bg


def draw(out, o):
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            x, y = o["x"] + i, o["y"] + j
            if v >= 0 and 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = v


def cells(o):
    return {(o["x"] + i, o["y"] + j) for j in range(o["h"]) for i in range(o["w"])}


def transition_function(state, action, frame):
    is_click = isinstance(action, dict)
    hy = hud_row(frame)
    bar = frame[hy] if hy < 64 else []
    if MEM["frame"] is not None and MEM["frame"] == frame:
        n = MEM["n"]
    else:
        z = sum(1 for v in bar if v == 3)
        n = 3 * z + 1 if z else 0
    out = [list(row) for row in frame]
    if not is_click:
        n += 1
        ring = next(o for o in state if o["type"] == "player")
        beam = next(o for o in state if o["type"] == "arm")
        blocks = [o for o in state if o["type"] == "block"]
        fr = field_rect(frame, hy)
        nring, nbeam, nblocks = step(ring, beam, blocks, action, fr)
        boxes = legend_boxes(frame, hy)
        order = {}
        for b in blocks:
            c = colour(b)
            if c in boxes:
                order[c] = min(p[0] for p in boxes[c])
        update_tags(nblocks, nbeam, order)
        covered = set()
        for o in [ring, beam] + blocks:
            covered |= cells(o)
        bg = background(frame, fr, hy, covered)
        for (x, y) in covered:
            if 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = bg(x, y)
        for o in sorted([nring, nbeam] + nblocks, key=lambda o: o["layer"]):
            draw(out, o)
        for b in nblocks:
            c = colour(b)
            if c not in boxes:
                continue
            xs = [p[0] for p in boxes[c]]
            ys = [p[1] for p in boxes[c]]
            x0, y0 = min(xs), min(ys)
            for yy in (y0 + 1, y0 + 2):
                for xx in (x0 + 1, x0 + 2):
                    out[yy][xx] = 0 if tagof(b) == "collected" else c
        if hy < 64:
            k = max(0, (n - 1) // 3)
            for i in range(min(k, 64)):
                if out[hy][63 - i] == 2:
                    out[hy][63 - i] = 3
    MEM["frame"] = out
    MEM["n"] = n
    return out
