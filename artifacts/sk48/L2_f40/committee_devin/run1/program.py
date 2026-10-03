# Crane/skewer: ring player + beam arm (h=2, w=1+6k, max 43) + 4x4 blocks tagged todo/next/collected.
# A1/A2: ring+beam move y-/+6, carrying blocks that overlap the beam rect; blocks hit get pushed (chains).
#   Any mover out of bounds (block y<2) cancels the whole move (that is the arm 'no_change' under A1).
# A4/A3: beam w+/-6; blocks lying fully on the beam follow the tip, swept blocks are pushed; a blocked
#   group (block x>48 or into ring) stays put while the beam still slides. Hypothesis: top/bottom bounds.
import copy

STEP, MAXW, MAX_BX, MIN_BY = 6, 43, 48, 2


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)],
            [2 if i % 3 == 0 else 1 for i in range(w)]]


def block_ok(r, ring):
    x, y, w, h = r
    return 0 <= x <= MAX_BX and MIN_BY <= y and y + h <= 64 and not overlap(r, rect(ring))


def push_group(blocks, movers, extra, dx, dy):
    """Move `movers` (block indices) by (dx,dy); blocks hit by moved blocks or by the
    `extra` rects (new positions of ring/beam) are pushed too. Returns set or None."""
    moved = set(movers)
    frontier = list(extra) + [shift(rect(blocks[i]), dx, dy) for i in moved]
    while frontier:
        r = frontier.pop()
        for i, b in enumerate(blocks):
            if i not in moved and overlap(rect(b), r):
                moved.add(i)
                frontier.append(shift(rect(b), dx, dy))
    return moved


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def fully_on(b, beam):
    return (overlap(rect(b), rect(beam)) and b["x"] >= beam["x"]
            and b["x"] + b["w"] <= beam["x"] + beam["w"])


def move_vertical(ring, beam, blocks, dy):
    nring = shift(rect(ring), 0, dy)
    nbeam = shift(rect(beam), 0, dy)
    carried = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
    moved = push_group(blocks, carried, [nring, nbeam], 0, dy)
    if nring[1] < 0 or nring[1] + nring[3] > 64:
        return
    if any(not (MIN_BY <= blocks[i]["y"] + dy and blocks[i]["y"] + dy + 4 <= 64) for i in moved):
        return
    ring["y"] += dy
    beam["y"] += dy
    for i in moved:
        blocks[i]["y"] += dy


def move_horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAXW:
        return
    onb = [i for i, b in enumerate(blocks) if fully_on(b, beam)]
    swept = []
    if dw > 0:
        tip = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        swept = [i for i, b in enumerate(blocks) if i not in onb and overlap(rect(b), tip)]
    moved = push_group(blocks, onb + swept, [], dw, 0)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    if moved and all(block_ok(shift(rect(blocks[i]), dw, 0), ring) for i in moved):
        for i in moved:
            blocks[i]["x"] += dw


def status(b):
    return b["tags"][-1]


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def update_tags(beam, blocks):
    tip = beam["x"] + beam["w"] - 1
    on = [b for b in blocks if overlap(rect(b), rect(beam))]

    def hooked(b):
        return b in on and b["x"] == tip - 4

    for b in blocks:
        if status(b) == "collected" and not hooked(b):
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
            return
    for b in blocks:
        if status(b) == "next" and hooked(b) and len(on) == 1:
            set_status(b, "collected")
            todo = [o for o in blocks if status(o) == "todo"]
            if todo:
                set_status(max(todo, key=lambda o: o["x"]), "next")
            return


def transition_function(state, action):
    out = copy.deepcopy(state)
    ring = next((o for o in out if o["type"] == "player"), None)
    beam = next((o for o in out if o["type"] == "arm"), None)
    blocks = [o for o in out if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return out
    if action in (1, 2):
        move_vertical(ring, beam, blocks, -STEP if action == 1 else STEP)
    elif action in (3, 4):
        move_horizontal(ring, beam, blocks, -STEP if action == 3 else STEP)
    else:
        return out
    update_tags(beam, blocks)
    return out
