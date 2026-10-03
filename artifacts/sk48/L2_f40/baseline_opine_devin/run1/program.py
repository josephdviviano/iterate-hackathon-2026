# Mechanics: crane ring (player) moves y+-6 (A1/A2) carrying blocks on the beam and pushing blocks it hits; A4/A3 grow/shrink the beam by 6.
# A hooked next/collected block (x == tip-4) drags all on-beam blocks; unhooked extension pushes swept next/collected blocks; a blocked
# horizontal chain leaves the beam changing alone, a blocked vertical move cancels. Walls = playfield bbox of colour 4 above the HUD bar.
# HUD bar row gains one 3 from the right per 3 non-click actions ((n-1)//3, n hidden, continuity-gated); legend swatch centre -> 0 while collected.
# Hypotheses unconfirmed: bar phase after a discontinuity (fallback n = 3*bars+1); background under the ring = rail pattern read from the frame.
import copy

_MEM = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def bar_row(frame):
    for y in range(len(frame)):
        if all(v in (2, 3) for v in frame[y]):
            return y
    return len(frame)


def playfield(frame, bar):
    cells = [(x, y) for y in range(bar) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def rail_columns(frame, bar, pf, covered):
    l, t, r, b = pf
    return [x for x in range(64) if not (l <= x <= r)
            and any(frame[y][x] in (2, 3) and (x, y) not in covered for y in range(bar))]


def background(x, y, pf, rails):
    l, t, r, b = pf
    if l <= x <= r and t <= y <= b:
        return 4
    if x in rails and t + 2 <= y <= b - 2:
        return 2 if (y - t - 2) % 6 < 2 else 3
    return 5


def block_ok(b, pf):
    l, t, r, bt = pf
    return b["x"] >= l and b["x"] + b["w"] - 1 <= r and b["y"] >= t and b["y"] + b["h"] - 1 <= bt


def tip(beam):
    return beam["x"] + beam["w"] - 1


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def status(b):
    for s in ("collected", "next", "todo"):
        if s in b["tags"]:
            return s
    return "todo"


def hooked(b, beam):
    return status(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == tip(beam) - 4


def push_chain(blocks, movers, solids, dx, dy):
    moved = set(movers)
    for i in movers:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
    changed = True
    while changed:
        changed = False
        frontier = [rect(blocks[i]) for i in moved] + solids
        for j, b in enumerate(blocks):
            if j in moved:
                continue
            if any(overlap(rect(b), f) for f in frontier):
                b["x"] += dx
                b["y"] += dy
                moved.add(j)
                changed = True
                break
    return moved


def step_vertical(ring, beam, blocks, dy, pf):
    l, t, r, b = pf
    nring, nbeam, nblocks = dict(ring), dict(beam), copy.deepcopy(blocks)
    nring["y"] += dy
    nbeam["y"] += dy
    if nring["y"] < t or nring["y"] + nring["h"] - 1 > b:
        return ring, beam, blocks
    carried = [i for i, bk in enumerate(blocks) if on_beam(bk, beam)]
    moved = push_chain(nblocks, carried, [rect(nring), rect(nbeam)], 0, dy)
    if any(not block_ok(nblocks[i], pf) for i in moved):
        return ring, beam, blocks
    return nring, nbeam, nblocks


def step_horizontal(ring, beam, blocks, dw, pf):
    l, t, r, b = pf
    nw = beam["w"] + dw
    if nw < 1 or beam["x"] + nw - 1 > r:
        return beam, blocks
    nbeam = dict(beam)
    nbeam["w"] = nw
    nblocks = copy.deepcopy(blocks)
    if any(hooked(bk, beam) for bk in blocks):
        movers = [i for i, bk in enumerate(blocks) if on_beam(bk, beam)]
        moved = push_chain(nblocks, movers, [], dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        movers = [i for i, bk in enumerate(blocks)
                  if status(bk) in ("next", "collected") and overlap(rect(bk), swept)]
        moved = push_chain(nblocks, movers, [], dw, 0)
    else:
        moved = set()
    if any(not block_ok(nblocks[i], pf) or overlap(rect(nblocks[i]), rect(ring)) for i in moved):
        return nbeam, blocks
    return nbeam, nblocks


def colour(b):
    for row in b["pixels"]:
        for v in row:
            if v >= 0:
                return v
    return None


def update_tags(blocks, beam, order):
    for b in blocks:
        b["tags"] = [s for s in b["tags"] if s not in ("collected", "next", "todo")] + [status(b)]

    def set_status(b, s):
        b["tags"] = [x for x in b["tags"] if x not in ("collected", "next", "todo")] + [s]

    nxt = [b for b in blocks if status(b) == "next"]
    if nxt and hooked(nxt[0], beam) and all(status(o) == "collected" for o in blocks
                                            if o is not nxt[0] and on_beam(o, beam)):
        set_status(nxt[0], "collected")
        todo = [b for b in blocks if status(b) == "todo"]
        todo.sort(key=lambda b: order.get(colour(b), 99))
        if todo:
            set_status(todo[0], "next")
        return
    if not any(hooked(b, beam) for b in blocks):
        for b in blocks:
            if status(b) == "collected" and not on_beam(b, beam):
                for o in blocks:
                    if status(o) == "next":
                        set_status(o, "todo")
                set_status(b, "next")
                break


def legend_boxes(frame, bar, colours):
    boxes = {}
    for c in colours:
        cells = [(x, y) for y in range(bar + 1, 64) for x in range(64) if frame[y][x] == c]
        if cells:
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            boxes[c] = (min(xs), min(ys), max(xs), max(ys))
    return boxes


def draw(out, o, pixels):
    for j, row in enumerate(pixels):
        for i, v in enumerate(row):
            x, y = o["x"] + i, o["y"] + j
            if v >= 0 and 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = v


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def transition_function(state, action, frame):
    out = [list(row) for row in frame]
    bar = bar_row(frame)
    is_click = not isinstance(action, int)
    threes = sum(1 for v in frame[bar] if v == 3) if bar < 64 else 0
    if _MEM["frame"] is not None and frame == _MEM["frame"]:
        n = _MEM["n"]
    else:
        n = 3 * threes + 1 if threes else 0
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is not None and beam is not None and not is_click:
        pf = playfield(frame, bar)
        covered = {(o["x"] + i, o["y"] + j) for o in [ring, beam] + blocks
                   for j in range(o["h"]) for i in range(o["w"])}
        rails = rail_columns(frame, bar, pf, covered)
        boxes = legend_boxes(frame, bar, [colour(b) for b in blocks])
        order = {c: bx[0] for c, bx in boxes.items()}
        nring, nbeam, nblocks = ring, beam, blocks
        if action in (1, 2):
            nring, nbeam, nblocks = step_vertical(ring, beam, blocks, -6 if action == 1 else 6, pf)
        elif action in (3, 4):
            nbeam, nblocks = step_horizontal(ring, beam, blocks, 6 if action == 4 else -6, pf)
        nblocks = copy.deepcopy(nblocks)
        update_tags(nblocks, nbeam, order)
        for o in [ring, beam] + blocks:
            for j in range(o["h"]):
                for i in range(o["w"]):
                    x, y = o["x"] + i, o["y"] + j
                    if 0 <= x < 64 and 0 <= y < 64:
                        out[y][x] = background(x, y, pf, rails)
        draw(out, nring, nring["pixels"])
        draw(out, nbeam, beam_pixels(nbeam["w"]))
        for b in nblocks:
            draw(out, b, b["pixels"])
        for b in nblocks:
            c = colour(b)
            if c not in boxes:
                continue
            x0, y0, x1, y1 = boxes[c]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            v = 0 if status(b) == "collected" else c
            for yy in (cy, cy + 1):
                for xx in (cx, cx + 1):
                    out[yy][xx] = v
    if not is_click:
        n += 1
    if bar < 64:
        k = max(0, (n - 1) // 3)
        for x in range(64):
            if frame[bar][x] in (2, 3):
                out[bar][x] = 3 if x >= 64 - k else 2
    _MEM["frame"] = out
    _MEM["n"] = n
    return [list(map(int, row)) for row in out]
