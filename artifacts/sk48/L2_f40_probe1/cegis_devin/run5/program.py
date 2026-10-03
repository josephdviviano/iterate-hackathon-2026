# Mechanics: crane ring (player) with beam (arm) at ring.x+5,y+2; A1/A2 move ring+beam y-/+6 carrying blocks on the
# beam and pushing blocks hit (chain), whole move cancelled if a block leaves y[2,58]. A4/A3 beam w +/-6 in [1,43]:
# a hooked next/collected block (x == tip-4) drags all on-beam blocks; otherwise extension pushes next/collected blocks
# in the swept columns; a group that would pass x>48 or into the ring stays put and the beam changes alone.
# Tags: hooked next with all other on-beam blocks collected -> collected, rightmost todo -> next; unhooked collected off beam reverts. Clicks no-op.
import copy

STEP = 6
WMAX = 43
XMAX = 48
YMIN, YMAX = 2, 58


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def status(b):
    for t in ("todo", "next", "collected"):
        if t in b["tags"]:
            return t
    return None


def set_status(b, s):
    b["tags"] = [t if t not in ("todo", "next", "collected") else s for t in b["tags"]]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def tip(beam):
    return beam["x"] + beam["w"] - 1


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); any other block overlapping a solid rect or a moved block is pushed too."""
    moved = set()
    queue = list(movers)
    for b in blocks:
        if b in movers:
            continue
        if any(overlap(rect(b), s) for s in solids):
            queue.append(b)
    while queue:
        b = queue.pop(0)
        if id(b) in moved:
            continue
        moved.add(id(b))
        b["x"] += dx
        b["y"] += dy
        for o in blocks:
            if id(o) not in moved and overlap(rect(o), rect(b)):
                queue.append(o)
    return [b for b in blocks if id(b) in moved]


def block_ok(b, ring):
    return YMIN <= b["y"] <= YMAX and b["x"] <= XMAX and b["x"] >= ring["x"] + ring["w"]


def hooked(blocks, beam):
    for b in blocks:
        if status(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == tip(beam) - 4:
            return b
    return None


def vertical(ring, beam, blocks, dy):
    if not (0 <= ring["y"] + dy <= YMAX):
        return False
    carried = [b for b in blocks if on_beam(b, beam)]
    ring["y"] += dy
    beam["y"] += dy
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(b, ring) for b in moved)


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if not (1 <= nw <= WMAX):
        return None
    h = hooked(blocks, beam)
    old_end = tip(beam)
    trial = copy.deepcopy(blocks)
    moved = []
    if h is not None:
        movers = [t for t, b in zip(trial, blocks) if on_beam(b, beam)]
        moved = push_chain(trial, movers, [], dw, 0)
    elif dw > 0:
        swept = (old_end + 1, beam["y"], dw, beam["h"])
        hit = [b for b in trial if status(b) in ("next", "collected") and overlap(rect(b), swept)]
        if hit:
            moved = push_chain(trial, hit, [], dw, 0)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    if all(block_ok(b, ring) for b in moved):
        for b, t in zip(blocks, trial):
            b["x"], b["y"] = t["x"], t["y"]
    return True


def update_tags(blocks, beam):
    h = hooked(blocks, beam)
    if h is not None and status(h) == "next":
        others = [b for b in blocks if b is not h and on_beam(b, beam)]
        if all(status(b) == "collected" for b in others):
            set_status(h, "collected")
            todo = [b for b in blocks if status(b) == "todo"]
            if todo:
                set_status(max(todo, key=lambda b: b["x"]), "next")
        return
    if h is None:
        off = [b for b in blocks if status(b) == "collected" and not on_beam(b, beam)]
        if off:
            for b in blocks:
                if status(b) == "next":
                    set_status(b, "todo")
            set_status(max(off, key=lambda b: b["x"]), "next")


def transition_function(state, action):
    new = copy.deepcopy(state)
    ring = next((o for o in new if o["type"] == "player"), None)
    beam = next((o for o in new if o["type"] == "arm"), None)
    blocks = [o for o in new if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return new
    if action in (1, 2):
        trial = copy.deepcopy(new)
        tr = next(o for o in trial if o["type"] == "player")
        tb = next(o for o in trial if o["type"] == "arm")
        tbl = [o for o in trial if o["type"] == "block"]
        if not vertical(tr, tb, tbl, -STEP if action == 1 else STEP):
            return new
        new, ring, beam, blocks = trial, tr, tb, tbl
    elif action in (3, 4):
        if horizontal(ring, beam, blocks, STEP if action == 4 else -STEP) is None:
            return new
    else:
        return new
    update_tags(blocks, beam)
    return new
