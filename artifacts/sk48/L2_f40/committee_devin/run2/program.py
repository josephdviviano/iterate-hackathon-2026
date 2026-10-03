# Mechanics: ring (player) at fixed x carries a dashed beam (arm, h=2, w=1+6k, at ring.x+5, ring.y+2).
# A1/A2: ring+beam y-/+6; blocks overlapping the beam are carried, blocks hit are pushed (chains);
# any blocked mover cancels the move. A4/A3: beam w+/-6. Unhooked extend pushes swept blocks
# (blocked chain -> beam slides under/through); hooked (block.x == tip-4) extend/retract moves on-beam blocks.
# Tags: 'next' hooked & alone on beam -> collected. Unconfirmed: bounds (block y>=2, x<=48, w<=43).
import copy

STEP = 6
MAX_W = 43
BLOCK_X_MAX = 48
BLOCK_Y_MIN = 2
BOARD = 64


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)],
            [2 if i % 3 == 0 else 1 for i in range(w)]]


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def tip(beam):
    return beam["x"] + beam["w"] - 1


def on_beam(beam, b):
    return overlap(rect(beam), rect(b))


def hooked_block(beam, blocks):
    for b in blocks:
        if on_beam(beam, b) and b["x"] + b["w"] == tip(beam):
            return b
    return None


def block_ok(b, dx, dy, ring):
    r = rect(b, dx, dy)
    if r[0] < 0 or r[0] > BLOCK_X_MAX or r[1] < BLOCK_Y_MIN or r[1] + r[3] > BOARD:
        return False
    return not overlap(r, rect(ring))


def push(movers, blocks, dx, dy, ring, extra_rects=()):
    """Grow the mover set by chain pushes; return the set or None if blocked."""
    movers = list(movers)
    ids = {id(b) for b in movers}
    changed = True
    while changed:
        changed = False
        pushers = [rect(m, dx, dy) for m in movers] + list(extra_rects)
        for b in blocks:
            if id(b) in ids:
                continue
            if any(overlap(p, rect(b)) for p in pushers):
                movers.append(b)
                ids.add(id(b))
                changed = True
    for m in movers:
        if not block_ok(m, dx, dy, ring):
            return None
    return movers


def apply(movers, dx, dy):
    for m in movers:
        m["x"] += dx
        m["y"] += dy


def vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0 or ny + ring["h"] > BOARD:
        return
    carried = [b for b in blocks if on_beam(beam, b)]
    movers = push(carried, blocks, 0, dy, ring,
                  (rect(beam, 0, dy), rect(ring, 0, dy)))
    if movers is None:
        return
    apply(movers, 0, dy)
    ring["y"] += dy
    beam["y"] += dy


def extend(ring, beam, blocks):
    if beam["w"] + STEP > MAX_W:
        return
    old_tip = tip(beam)
    h = hooked_block(beam, blocks)
    if h is not None:
        riding = [b for b in blocks if on_beam(beam, b) and b["x"] + b["w"] <= old_tip + 1]
        movers = push(riding, blocks, STEP, 0, ring)
        if movers is not None:
            apply(movers, STEP, 0)
    else:
        sweep = (old_tip + 1, beam["y"], STEP, beam["h"])
        swept = [b for b in blocks if overlap(sweep, rect(b))]
        movers = push(swept, blocks, STEP, 0, ring)
        if movers is not None:
            apply(movers, STEP, 0)
    beam["w"] += STEP


def retract(ring, beam, blocks):
    if beam["w"] - STEP < 1:
        return
    h = hooked_block(beam, blocks)
    if h is not None:
        riding = [b for b in blocks if on_beam(beam, b)]
        movers = push(riding, blocks, -STEP, 0, ring)
        if movers is not None:
            apply(movers, -STEP, 0)
    beam["w"] -= STEP


def set_status(b, status):
    b["tags"] = [t for t in b["tags"] if t not in ("todo", "next", "collected")] + [status]


def status(b):
    for t in ("todo", "next", "collected"):
        if t in b["tags"]:
            return t
    return None


def update_tags(beam, blocks):
    h = hooked_block(beam, blocks)
    alone = h is not None and all(b is h or not on_beam(beam, b) for b in blocks)
    for b in blocks:
        if status(b) == "collected" and not (alone and b is h):
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
    if alone and status(h) == "next":
        set_status(h, "collected")
        todo = [b for b in blocks if status(b) == "todo"]
        if todo:
            set_status(max(todo, key=lambda b: (b["x"], -b["y"])), "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o.get("type") == "player"), None)
    beam = next((o for o in state if o.get("type") == "arm"), None)
    blocks = [o for o in state if o.get("type") == "block"]
    if ring is None or beam is None or isinstance(action, dict):
        return state
    if action == 1:
        vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        vertical(ring, beam, blocks, STEP)
    elif action == 4:
        extend(ring, beam, blocks)
    elif action == 3:
        retract(ring, beam, blocks)
    else:
        return state
    beam["pixels"] = beam_pixels(beam["w"])
    update_tags(beam, blocks)
    return state
