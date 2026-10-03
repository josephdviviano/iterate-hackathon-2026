# Mechanics: ring (player) + dashed beam (arm, x=ring.x+5, y=ring.y+2, w=1+6k<=43) + 4x4 blocks. A1/A2 move ring+beam y-/+6,
# carrying blocks overlapping the beam and pushing (chain) blocks hit by new ring/beam rects; any block y outside [2,58] cancels all.
# A4/A3 beam w+/-6: if a next/collected block sits at the tip (x==beam.x+w-5) all on-beam blocks follow; else extend pushes
# next/collected blocks in the swept columns (chain). A moved block with x>48 or inside the ring keeps blocks put, beam changes alone.
# Tags: hooked next with all other on-beam blocks collected -> collected, rightmost todo -> next; unhooked collected off beam -> next.
import copy

BLOCK_Y = (2, 58)
BLOCK_XMAX = 48
RING_Y = (0, 58)
STEP = 6
WMAX = 43


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def tag(b):
    return b["tags"][-1]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked_block(blocks, beam):
    tip = beam["x"] + beam["w"] - 5
    for b in blocks:
        if tag(b) in ("next", "collected") and b["x"] == tip and on_beam(b, beam):
            return b
    return None


def push_chain(blocks, movers, solids, dx, dy, pushable=None):
    """Move movers by (dx,dy); blocks hit by solids (if pushable) or by moved blocks are pushed too."""
    moved = set(id(b) for b in movers)
    for b in movers:
        b["x"] += dx
        b["y"] += dy
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if id(b) in moved:
                continue
            hit = any(overlap(rect(b), rect(m)) for m in blocks if id(m) in moved)
            if not hit and (pushable is None or pushable(b)):
                hit = any(overlap(rect(b), s) for s in solids)
            if hit:
                moved.add(id(b))
                b["x"] += dx
                b["y"] += dy
                changed = True
    return [b for b in blocks if id(b) in moved]


def block_ok(b, ring):
    return BLOCK_Y[0] <= b["y"] <= BLOCK_Y[1] and b["x"] <= BLOCK_XMAX and b["x"] >= ring["x"] + ring["w"]


def set_tag(b, t):
    b["tags"] = b["tags"][:-1] + [t]


def update_tags(blocks, beam):
    h = hooked_block(blocks, beam)
    if h is not None:
        others = [b for b in blocks if b is not h and on_beam(b, beam)]
        if tag(h) == "next" and all(tag(b) == "collected" for b in others):
            set_tag(h, "collected")
            todo = [b for b in blocks if tag(b) == "todo"]
            if todo:
                set_tag(max(todo, key=lambda b: (b["x"], -b["y"])), "next")
        return
    off = [b for b in blocks if tag(b) == "collected" and not on_beam(b, beam)]
    if off:
        for b in blocks:
            if tag(b) == "next":
                set_tag(b, "todo")
        set_tag(max(off, key=lambda b: b["x"]), "next")


def vertical(ring, beam, blocks, dy):
    if not (RING_Y[0] <= ring["y"] + dy <= RING_Y[1]):
        return False
    carried = [b for b in blocks if on_beam(b, beam)]
    ring["y"] += dy
    beam["y"] += dy
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(b, ring) for b in moved)


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if not (1 <= nw <= WMAX):
        return False
    h = hooked_block(blocks, beam)
    old_end = beam["x"] + beam["w"]
    trial = copy.deepcopy(blocks)
    if h is not None:
        movers = [t for t, b in zip(trial, blocks) if on_beam(b, beam)]
        moved = push_chain(trial, movers, [], dw, 0)
    elif dw > 0:
        swept = (old_end, beam["y"], dw, beam["h"])
        moved = push_chain(trial, [], [swept], dw, 0,
                           pushable=lambda b: tag(b) in ("next", "collected"))
    else:
        moved = []
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    if all(block_ok(b, ring) for b in moved):
        for b, t in zip(blocks, trial):
            b["x"], b["y"] = t["x"], t["y"]
    return True


def transition_function(state, action):
    out = copy.deepcopy(state)
    ring = next((o for o in out if o.get("type") == "player"), None)
    beam = next((o for o in out if o.get("type") == "arm"), None)
    blocks = [o for o in out if o.get("type") == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return out
    trial = copy.deepcopy(out)
    tring = next(o for o in trial if o.get("type") == "player")
    tbeam = next(o for o in trial if o.get("type") == "arm")
    tblocks = [o for o in trial if o.get("type") == "block"]
    if action in (1, 2):
        ok = vertical(tring, tbeam, tblocks, -STEP if action == 1 else STEP)
    elif action in (3, 4):
        ok = horizontal(tring, tbeam, tblocks, STEP if action == 4 else -STEP)
    else:
        ok = False
    if not ok:
        return out
    update_tags(tblocks, tbeam)
    return trial
