# Crane/skewer game. ring (player 6x6) moves with its beam (arm, h=2, w=1+6k<=43, at ring.x+5,y+2).
# A1/A2: ring+beam y-/+6 carrying blocks overlapping the beam; blocks hit get pushed (chains); any
# block past y<2 / bottom cancels the whole move. A4/A3: beam w+/-6; blocks fully on the beam (+ blocks
# swept by the tip on extend) shift +/-6 with chain pushes; if any would leave x<=48 or hit the ring the
# blocks stay and the beam still changes. Tags: 'next' block hooked (x==tip-4) alone on beam -> collected.
import copy

GRID = 64
BLOCK_MAX_X = 48
BLOCK_MIN_Y = 2
MAX_W = 43
STEP = 6


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    return b["tags"][-1]


def on_beam_fully(b, beam):
    bx, by, bw, bh = rect(beam)
    return (b["x"] >= bx and b["x"] + b["w"] <= bx + bw
            and b["y"] < by + bh and by < b["y"] + b["h"])


def push_chain(blocks, movers, dx, dy, extra_rects=()):
    """Move `movers` by (dx,dy), pushing any block they (or extra_rects) hit. Returns moved set."""
    moved = set(movers)
    hitters = list(extra_rects)
    while True:
        new_rects = [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, 4, 4) for i in moved] + hitters
        added = [j for j in range(len(blocks)) if j not in moved
                 and any(overlap(rect(blocks[j]), r) for r in new_rects)]
        if not added:
            return moved
        moved.update(added)


def vertical(ring, beam, blocks, dy):
    carried = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
    new_ring = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
    new_beam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    if new_ring[1] < 0 or new_ring[1] + new_ring[3] > GRID:
        return
    moved = push_chain(blocks, carried, 0, dy, (new_ring, new_beam))
    for i in moved:
        ny = blocks[i]["y"] + dy
        if ny < BLOCK_MIN_Y or ny + blocks[i]["h"] > GRID - BLOCK_MIN_Y:
            return
    ring["y"] += dy
    beam["y"] += dy
    for i in moved:
        blocks[i]["y"] += dy


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return
    tip_old = beam["x"] + beam["w"]
    movers = [i for i, b in enumerate(blocks) if on_beam_fully(b, beam)]
    if dw > 0:
        sweep = (tip_old, beam["y"], dw, beam["h"])
        movers += [i for i, b in enumerate(blocks) if i not in movers and overlap(rect(b), sweep)]
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    if not movers:
        return
    moved = push_chain(blocks, movers, dw, 0)
    for i in moved:
        nr = (blocks[i]["x"] + dw, blocks[i]["y"], 4, 4)
        if nr[0] > BLOCK_MAX_X or nr[0] < 0 or overlap(nr, rect(ring)):
            return
    for i in moved:
        blocks[i]["x"] += dw


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def update_tags(beam, blocks):
    tip = beam["x"] + beam["w"] - 1
    on = [b for b in blocks if overlap(rect(b), rect(beam))]

    def hooked(b):
        return len(on) == 1 and on[0] is b and b["x"] == tip - 4
    for b in blocks:
        if status(b) == "collected" and not hooked(b):
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
            return
    for b in blocks:
        if status(b) == "next" and hooked(b):
            set_status(b, "collected")
            todo = [o for o in blocks if status(o) == "todo"]
            if todo:
                set_status(max(todo, key=lambda o: (o["x"], -o["y"])), "next")
            return


def transition_function(state, action):
    out = copy.deepcopy(state)
    ring = next((o for o in out if o["type"] == "player"), None)
    beam = next((o for o in out if o["type"] == "arm"), None)
    blocks = [o for o in out if o["type"] == "block"]
    if ring is None or beam is None or isinstance(action, dict):
        return out
    if action == 1:
        vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        vertical(ring, beam, blocks, STEP)
    elif action == 4:
        horizontal(ring, beam, blocks, STEP)
    elif action == 3:
        horizontal(ring, beam, blocks, -STEP)
    else:
        return out
    update_tags(beam, blocks)
    return out
