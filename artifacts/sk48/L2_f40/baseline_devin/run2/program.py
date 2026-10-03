# Mechanics: ring(player)+beam(arm) crane. A1/A2 move ring+beam by -/+6 in y, carrying blocks the beam overlaps and pushing
# blocks hit by the moving ring/beam (chain); any push out of bounds/into the ring blocks the whole move. A4/A3 extend/
# retract the beam by 6 (w in 1..43): blocks fully on the beam move with the tip, blocks in the swept area are pushed;
# if that block group is blocked the beam slides under/out alone. Tags: the 'next' block becomes 'collected' when it is the
# only block on the beam; un-collected when it leaves the beam. Hypotheses: bounds [2,56) and the 'alone' gate are guesses.
import copy

STEP = 6
WMAX = 43
LO, HI = 2, 56
PAT0, PAT1 = (1, 2, 1), (2, 1, 1)


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["x"] + dx + o["w"], o["y"] + dy + o["h"])


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def in_bounds(r):
    return r[0] >= LO and r[1] >= LO and r[2] <= HI and r[3] <= HI


def on_beam(blk, beam):
    """Block lies in the beam's rows and entirely within its x-span."""
    b, r = rect(beam), rect(blk)
    return r[1] < b[3] and b[1] < r[3] and r[0] >= b[0] and r[2] <= b[2]


def push_group(blocks, seeds, dx, dy, obstacles):
    """Close the moving set under block-block pushing; return it, or None if blocked."""
    moving = set(seeds)
    frontier = list(seeds)
    while frontier:
        i = frontier.pop()
        nr = rect(blocks[i], dx, dy)
        if not in_bounds(nr) or any(overlap(nr, ob) for ob in obstacles):
            return None
        for j, other in enumerate(blocks):
            if j not in moving and overlap(nr, rect(other)):
                moving.add(j)
                frontier.append(j)
    return moving


def set_beam_w(beam, w):
    beam["w"] = w
    beam["pixels"] = [[PAT0[i % 3] for i in range(w)], [PAT1[i % 3] for i in range(w)]]


def vertical(ring, beam, blocks, dy):
    if not in_bounds(rect(ring, 0, dy)) or not in_bounds(rect(beam, 0, dy)):
        return
    br = rect(beam)
    seeds = {i for i, b in enumerate(blocks) if overlap(rect(b), br)}
    nring, nbeam = rect(ring, 0, dy), rect(beam, 0, dy)
    seeds |= {i for i, b in enumerate(blocks) if overlap(rect(b), nring) or overlap(rect(b), nbeam)}
    moving = push_group(blocks, seeds, 0, dy, [])
    if moving is None:
        return
    if any(overlap(rect(blocks[i], 0, dy), nring) for i in moving if i not in seeds):
        return
    ring["y"] += dy
    beam["y"] += dy
    for i in moving:
        blocks[i]["y"] += dy


def horizontal(ring, beam, blocks, dw):
    w = beam["w"] + dw
    if w < 1 or w > WMAX:
        return
    tip = beam["x"] + beam["w"]
    seeds = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
    if dw > 0:
        swept = (tip, beam["y"], tip + dw, beam["y"] + beam["h"])
        seeds |= {i for i, b in enumerate(blocks) if overlap(rect(b), swept)}
    moving = push_group(blocks, seeds, dw, 0, [rect(ring)]) if seeds else set()
    set_beam_w(beam, w)
    for i in moving or ():
        blocks[i]["x"] += dw


def status(b):
    return b["tags"][-1]


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def update_tags(beam, blocks):
    seq = list(reversed(blocks))
    for b in seq:
        if status(b) == "collected" and not on_beam(b, beam):
            for o in seq:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
    nxt = [b for b in seq if status(b) == "next"]
    covered = [b for b in blocks if on_beam(b, beam)]
    if nxt and covered == [nxt[0]]:
        set_status(nxt[0], "collected")
        k = seq.index(nxt[0])
        for o in seq[k + 1:]:
            if status(o) == "todo":
                set_status(o, "next")
                break


def transition_function(state, action):
    st = copy.deepcopy(state)
    ring = next((o for o in st if o.get("type") == "player"), None)
    beam = next((o for o in st if o.get("type") == "arm"), None)
    blocks = [o for o in st if o.get("type") == "block"]
    if ring is None or beam is None or isinstance(action, dict):
        return st
    if action == 1:
        vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        vertical(ring, beam, blocks, STEP)
    elif action == 4:
        horizontal(ring, beam, blocks, STEP)
    elif action == 3:
        horizontal(ring, beam, blocks, -STEP)
    else:
        return st
    update_tags(beam, blocks)
    return st
