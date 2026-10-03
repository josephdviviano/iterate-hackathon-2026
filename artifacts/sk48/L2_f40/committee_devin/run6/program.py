# Mechanics: crane/skewer. ring (player) + beam (arm, at ring.x+5, ring.y+2, h=2, w=1+6k<=43) + 4x4 blocks.
# A1/A2: ring+beam y-/+6, blocks overlapping the beam are carried, blocks hit get pushed (chain); any block y<2 cancels.
# A4/A3: beam w+/-6. Extend pushes swept blocks (chain, x<=48, else beam slides alone); hooked (next/collected block at tip)
# drags all on-beam blocks with the tip; a hooked retract into the ring lets the beam shrink alone. Clicks/A5/A7: no-op.
# Tags: next block hooked & alone on beam -> collected, rightmost todo -> next; unhooked collected reverts. Hypothesis: bounds y<=58.
import copy

STEP, MAXW, MINY, MAXY, MAXX = 6, 43, 2, 58, 48


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def status(b):
    return b["tags"][-1]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked_block(blocks, beam):
    for b in blocks:
        if status(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == beam["x"] + beam["w"] - 5:
            return b
    return None


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); blocks hit by solids or moved blocks are pushed too. Returns new rects or None."""
    pos = {id(b): rect(b) for b in blocks}
    moved = set()
    frontier = list(movers)
    for b in movers:
        moved.add(id(b))
    pend = [(r) for r in solids]
    while True:
        for b in frontier:
            x, y, w, h = pos[id(b)]
            pos[id(b)] = (x + dx, y + dy, w, h)
        hitters = pend + [pos[id(b)] for b in frontier]
        pend, frontier = [], []
        for b in blocks:
            if id(b) not in moved and any(overlap(pos[id(b)], r) for r in hitters):
                moved.add(id(b))
                frontier.append(b)
        if not frontier:
            break
    for b in blocks:
        if id(b) in moved:
            x, y, w, h = pos[id(b)]
            if y < MINY or y > MAXY or x > MAXX or x < 0:
                return None
    return pos


def vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0 or ny + ring["h"] > 64:
        return False
    carried = [b for b in blocks if on_beam(b, beam)]
    nring = (ring["x"], ny, ring["w"], ring["h"])
    nbeam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    pos = push_chain(blocks, carried, [nring, nbeam], 0, dy)
    if pos is None:
        return False
    ring["y"], beam["y"] = ny, beam["y"] + dy
    for b in blocks:
        b["x"], b["y"] = pos[id(b)][0], pos[id(b)][1]
    return True


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAXW:
        return
    hook = hooked_block(blocks, beam)
    ringr = rect(ring)
    pos = None
    if hook is not None:
        movers = [b for b in blocks if on_beam(b, beam)]
        pos = push_chain(blocks, movers, [], dw, 0)
        if pos is not None and any(overlap(pos[id(b)], ringr) for b in blocks):
            pos = None
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        pos = push_chain(blocks, [], [swept], dw, 0)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    if pos is not None:
        for b in blocks:
            b["x"], b["y"] = pos[id(b)][0], pos[id(b)][1]


def retag(blocks, beam):
    hook = hooked_block(blocks, beam)
    alone = hook is not None and [b for b in blocks if on_beam(b, beam)] == [hook]
    for b in blocks:
        if status(b) == "collected" and not (alone and b is hook):
            for o in blocks:
                if status(o) == "next":
                    o["tags"][-1] = "todo"
            b["tags"][-1] = "next"
    if alone and status(hook) == "next":
        hook["tags"][-1] = "collected"
        todo = [b for b in blocks if status(b) == "todo"]
        if todo:
            max(todo, key=lambda b: b["x"])["tags"][-1] = "next"


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is None or beam is None or isinstance(action, dict):
        return state
    if action in (1, 2):
        vertical(ring, beam, blocks, -STEP if action == 1 else STEP)
    elif action in (3, 4):
        horizontal(ring, beam, blocks, STEP if action == 4 else -STEP)
    else:
        return state
    retag(blocks, beam)
    return state
