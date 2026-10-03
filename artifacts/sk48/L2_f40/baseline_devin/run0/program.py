# Mechanics: ring(player) moves y+-6 on ACTION1/2 dragging its beam(arm, at ring.x+5, ring.y+2); ACTION4/3 grow/shrink beam w by 6 (min 1).
# Blocks inside the beam rect ride vertical moves; blocks hit by the moved ring/beam/blocks are pushed (all-or-nothing, bounds/ring block).
# Extending: the tip passes through 'todo' blocks but pushes the active ('next') block to tip+2; if it can't move the tip slides under -> hooked (x == tip-4).
# Hooked: every block in the beam rect follows the tip; blocked retract unhooks, blocked extend is a no-op. Hooked alone => 'collected', next = rightmost todo.
# Unconfirmed: field bounds (x<=53, y>=4 guessed), unhooked retract (assumed no drag), next-block order (rightmost todo), clicks/5/7 = no-op.
import copy

STEP, XMAX, YMIN, YMAX = 6, 53, 4, 63


def rect(o):
    return (o["x"], o["y"], o["x"] + o["w"] - 1, o["y"] + o["h"] - 1)


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy)


def in_bounds(r):
    return r[0] >= 0 and r[2] <= XMAX and r[1] >= YMIN and r[3] <= YMAX


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    for t in ("todo", "next", "collected"):
        if t in b["tags"]:
            return t
    return None


def set_status(b, s):
    b["tags"] = [s if t in ("todo", "next", "collected") else t for t in b["tags"]]


def tip(beam):
    return beam["x"] + beam["w"] - 1


def on_beam(beam, blocks):
    br = rect(beam)
    return [b for b in blocks if overlap(rect(b), br)]


def hooked(beam, blocks):
    r = rect(beam)
    for b in blocks:
        if status(b) in ("next", "collected") and b["x"] == tip(beam) - 4 and overlap(rect(b), r):
            return b
    return None


def group_move(movers, blocks, ring, dx, dy, extra=()):
    """Move blocks in movers (plus pushed chain) by dx,dy; return moved set or None if blocked."""
    moving = list(movers)
    ids = {id(b) for b in moving}
    sources = [shift(rect(b), dx, dy) for b in moving] + list(extra)
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if id(b) in ids:
                continue
            if any(overlap(rect(b), s) for s in sources):
                moving.append(b)
                ids.add(id(b))
                sources.append(shift(rect(b), dx, dy))
                changed = True
    ring_r = rect(ring) if ring is not None else None
    for b in moving:
        nr = shift(rect(b), dx, dy)
        if not in_bounds(nr) or (ring_r is not None and overlap(nr, ring_r)):
            return None
    return moving


def update_tags(beam, blocks):
    h = hooked(beam, blocks)
    for b in blocks:
        if status(b) == "collected" and b is not h:
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
    if h is not None and status(h) == "next" and on_beam(beam, blocks) == [h]:
        set_status(h, "collected")
        todo = [b for b in blocks if status(b) == "todo"]
        if todo:
            set_status(max(todo, key=lambda b: (b["x"], -b["y"])), "next")


def transition_function(state, action):
    st = copy.deepcopy(state)
    ring = next((o for o in st if o.get("type") == "player"), None)
    beam = next((o for o in st if o.get("type") == "arm"), None)
    blocks = [o for o in st if o.get("type") == "block"]
    a = action.get("action_id") if isinstance(action, dict) else action
    if ring is None or beam is None:
        return st
    if a in (1, 2):
        dy = -STEP if a == 1 else STEP
        nring, nbeam = shift(rect(ring), 0, dy), shift(rect(beam), 0, dy)
        if not in_bounds(nring) or not in_bounds(nbeam):
            return st
        carried = on_beam(beam, blocks)
        moving = group_move(carried, blocks, None, 0, dy, extra=(nring, nbeam))
        if moving is None:
            return st
        ring["y"] += dy
        beam["y"] += dy
        for b in moving:
            b["y"] += dy
    elif a in (3, 4):
        h = hooked(beam, blocks)
        dx = STEP if a == 4 else -STEP
        nw = beam["w"] + dx
        if nw < 1 or beam["x"] + nw - 1 > XMAX:
            return st
        if h is not None:
            moving = group_move(on_beam(beam, blocks), blocks, ring, dx, 0)
            if moving is None:
                if a == 4:
                    return st
            else:
                for b in moving:
                    b["x"] += dx
        elif a == 4:
            r = rect(beam)
            ntip = r[2] + dx
            sweep = (r[2] + 1, r[1], ntip, r[3])
            for b in blocks:
                if status(b) == "next" and overlap(rect(b), sweep):
                    ddx = ntip + 2 - b["x"]
                    others = [o for o in blocks if o is not b]
                    nr = shift(rect(b), ddx, 0)
                    if in_bounds(nr) and not any(overlap(nr, rect(o)) for o in others):
                        b["x"] += ddx
        beam["w"] = nw
        beam["pixels"] = beam_pixels(nw)
        update_tags(beam, blocks)
    return st
