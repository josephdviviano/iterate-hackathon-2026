# Crane/skewer: ring (player) carries a dashed beam (arm, h=2, w=1+6k<=43) at ring.x+5, ring.y+2; blocks are 4x4.
# A1/A2: ring+beam y-/+6; blocks overlapping the beam are carried, blocks hit get chain-pushed; any blocked block cancels all.
# A4/A3: beam w+/-6; a next/collected block hooked at tip (x==end-5) moves all on-beam blocks, else extend chain-pushes newly hit blocks; blocked => beam slides.
# Tags: hooked 'next' block alone on beam (+collected) -> collected, rightmost todo -> next; collected leaving beam -> next (old next -> todo).
# Unconfirmed: bounds (block y>=2, x<=48) and multi-collected reverting; arm A4 no-op = w already at max 43 / hooked block at x max.
import copy

STEP = 6
MAX_W = 43
BLOCK_MAX_X = 48
BLOCK_MIN_Y = 2


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def block_ok(r, ring_r):
    return r[0] >= 0 and r[0] <= BLOCK_MAX_X and r[1] >= BLOCK_MIN_Y and not overlap(r, ring_r)


def push(blocks, movers, dx, dy, ring_r, extra=None):
    """Move `movers` by (dx,dy); blocks they (or `extra` rect) hit get pushed too. Returns new rects or None."""
    pos = {n: rect(b) for n, b in blocks.items()}
    moving = set(movers)
    if extra is not None:
        moving |= {n for n, r in pos.items() if overlap(r, extra)}
    changed = True
    while changed:
        changed = False
        for n in list(moving):
            r = pos[n]
            nr = (r[0] + dx, r[1] + dy, r[2], r[3])
            for m, q in pos.items():
                if m not in moving and overlap(nr, q):
                    moving.add(m)
                    changed = True
    new = dict(pos)
    for n in moving:
        r = pos[n]
        new[n] = (r[0] + dx, r[1] + dy, r[2], r[3])
        if not block_ok(new[n], ring_r):
            return None
    return new


def on_beam(blocks, beam):
    br = rect(beam)
    return {n for n, b in blocks.items() if overlap(rect(b), br)}


def hooked(blocks, beam):
    end = beam["x"] + beam["w"]
    return [n for n in on_beam(blocks, beam)
            if blocks[n]["x"] == end - 5 and blocks[n]["tags"][-1] != "todo"]


def apply(blocks, new):
    for n, r in new.items():
        blocks[n]["x"], blocks[n]["y"] = r[0], r[1]


def vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0:
        return
    carried = on_beam(blocks, beam)
    ring_r = (ring["x"], ny, ring["w"], ring["h"])
    nbeam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    new = push(blocks, carried, 0, dy, ring_r, extra=None)
    if new is None:
        return
    # blocks the beam itself runs into at its new position
    hit = {n for n in blocks if n not in carried and overlap(new[n], nbeam)}
    if hit:
        tmp = {n: dict(blocks[n], x=new[n][0], y=new[n][1]) for n in blocks}
        new2 = push(tmp, hit, 0, dy, ring_r)
        if new2 is None:
            return
        new = new2
    ring["y"] = ny
    beam["y"] += dy
    apply(blocks, new)


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return
    ring_r = rect(ring)
    hk = hooked(blocks, beam)
    if hk:
        new = push(blocks, on_beam(blocks, beam), dw, 0, ring_r)
        if new is None:
            if dw > 0:
                return
        else:
            apply(blocks, new)
    elif dw > 0:
        old = rect(beam)
        nb = (beam["x"], beam["y"], nw, beam["h"])
        hit = {n for n, b in blocks.items() if overlap(rect(b), nb) and not overlap(rect(b), old)}
        if hit:
            new = push(blocks, hit, dw, 0, ring_r)
            if new is not None:
                apply(blocks, new)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)


def set_tag(b, t):
    b["tags"] = b["tags"][:-1] + [t]


def update_tags(blocks, beam):
    tag = lambda n: blocks[n]["tags"][-1]
    ob = on_beam(blocks, beam)
    for n in blocks:
        if tag(n) == "collected" and n not in ob:
            for m in blocks:
                if tag(m) == "next":
                    set_tag(blocks[m], "todo")
            set_tag(blocks[n], "next")
    collected = {n for n in blocks if tag(n) == "collected"}
    for h in hooked(blocks, beam):
        if tag(h) == "next" and ob <= collected | {h}:
            set_tag(blocks[h], "collected")
            todo = [n for n in blocks if tag(n) == "todo"]
            if todo:
                nxt = max(todo, key=lambda n: (blocks[n]["x"], -blocks[n]["y"]))
                set_tag(blocks[nxt], "next")


def transition_function(state, action):
    st = copy.deepcopy(state)
    ring = next((o for o in st if o["type"] == "player"), None)
    beam = next((o for o in st if o["type"] == "arm"), None)
    if ring is None or beam is None:
        return st
    blocks = {o["name"]: o for o in st if o["type"] == "block"}
    a = action if isinstance(action, int) else action.get("action_id")
    if a == 1:
        vertical(ring, beam, blocks, -STEP)
    elif a == 2:
        vertical(ring, beam, blocks, STEP)
    elif a == 4:
        horizontal(ring, beam, blocks, STEP)
    elif a == 3:
        horizontal(ring, beam, blocks, -STEP)
    else:
        return st
    update_tags(blocks, beam)
    return st
