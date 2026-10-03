# Crane/skewer: ring (player) at fixed x, beam (arm) from ring at y+2, h=2, w=1+6k (max 43); 4x4 blocks.
# A1/A2: ring+beam move y-/+6 carrying blocks overlapping the beam; anything they hit is pushed (chain);
#   any block/ring out of bounds cancels the whole move. A4/A3: beam w+/-6. If a block sits at the tip
#   (x == tip-4) it is hooked: all blocks on the beam move with the tip; otherwise extend pushes swept blocks.
#   Blocked horizontal pushes (ring or right wall) let the beam slide alone. Tags: hooked 'next' block alone on beam -> collected, rightmost todo -> next; unhooked collected reverts. Hypothesis: bounds (block y>=2, x+4<=56) unconfirmed.
import copy

STEP = 6
MAX_W = 43


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def block_in_bounds(r, ring_r):
    x, y, w, h = r
    return y >= 2 and y + h <= 62 and x + w <= 56 and not overlap(r, ring_r)


def push(blocks, movers, dx, dy, pushers, ring_r):
    """Move `movers` (indices) by (dx,dy); blocks hit by moved rects or `pushers` are pushed too.
    Returns {index: new rect} or None if blocked."""
    moved = {}
    queue = list(movers)
    for i in queue:
        moved[i] = None
    def shifted(i):
        x, y, w, h = rect(blocks[i])
        return (x + dx, y + dy, w, h)
    hit_by = list(pushers)
    while True:
        for i in list(moved):
            if moved[i] is None:
                moved[i] = shifted(i)
                hit_by.append(moved[i])
        new = [j for j in range(len(blocks)) if j not in moved and any(overlap(rect(blocks[j]), r) for r in hit_by)]
        if not new:
            break
        for j in new:
            moved[j] = None
    for r in moved.values():
        if not block_in_bounds(r, ring_r):
            return None
    return moved


def transition_function(state, action):
    st = copy.deepcopy(state)
    ring = next((o for o in st if o.get("type") == "player"), None)
    beam = next((o for o in st if o.get("type") == "arm"), None)
    if ring is None or beam is None or isinstance(action, dict) or action not in (1, 2, 3, 4):
        return st
    blocks = [o for o in st if o.get("type") == "block"]
    tip = beam["x"] + beam["w"] - 1
    on_beam = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
    hooked = [i for i in on_beam if blocks[i]["x"] == tip - blocks[i]["w"]]

    if action in (1, 2):
        dy = -STEP if action == 1 else STEP
        new_ring = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
        new_beam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
        if new_ring[1] < 0 or new_ring[1] + new_ring[3] > 64:
            return st
        moved = push(blocks, on_beam, 0, dy, [new_ring, new_beam], (-99, -99, 0, 0))
        if moved is None:
            return st
        ring["y"], beam["y"] = new_ring[1], new_beam[1]
        for i, r in moved.items():
            blocks[i]["x"], blocks[i]["y"] = r[0], r[1]
    else:
        dw = STEP if action == 4 else -STEP
        new_w = beam["w"] + dw
        if new_w < 1 or new_w > MAX_W:
            return st
        ring_r = rect(ring)
        moved = None
        if hooked:
            moved = push(blocks, on_beam, dw, 0, [], ring_r)
        elif dw > 0:
            sweep = (tip + 1, beam["y"], dw, beam["h"])
            hit = [i for i, b in enumerate(blocks) if overlap(rect(b), sweep)]
            if hit:
                moved = push(blocks, hit, dw, 0, [], ring_r)
        beam["w"] = new_w
        beam["pixels"] = beam_pixels(new_w)
        for i, r in (moved or {}).items():
            blocks[i]["x"] = r[0]
    update_tags(blocks, beam)
    return st


def status(b):
    return b["tags"][-1]


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def update_tags(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    on = [b for b in blocks if overlap(rect(b), rect(beam))]
    for b in blocks:
        if status(b) == "collected" and b not in on:
            for n in blocks:
                if status(n) == "next":
                    set_status(n, "todo")
            set_status(b, "next")
    nxt = [b for b in blocks if status(b) == "next"]
    if nxt and on == nxt and nxt[0]["x"] == tip - nxt[0]["w"]:
        set_status(nxt[0], "collected")
        todo = [b for b in blocks if status(b) == "todo"]
        if todo:
            set_status(max(todo, key=lambda b: b["x"]), "next")
