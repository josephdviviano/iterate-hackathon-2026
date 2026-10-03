# Mechanics: ring+beam (arm h=2, w=1+6k<=43) + 4x4 blocks. A1/A2 move ring+beam y-/+6, carrying on-beam blocks and pushing hit ones (chain); any block leaving y[2,58] cancels the move.
# A4/A3: beam w+/-6. A hooked block (next/collected at x==beam.x+w-5) drags all on-beam blocks; otherwise extension pushes swept blocks; a blocked group (x>48, ring, chain) lets the beam change alone.
# Tags: a hooked 'next' block whose other on-beam blocks are ALL collected becomes collected (not only when alone on the beam), and the rightmost todo becomes next.
# A collected block that leaves the beam reverts to next and the current next reverts to todo. Clicks and A5/A7 are no-ops; stateless.
# Unconfirmed: ring vertical bounds, and which collected block reverts if several leave the beam at once.
import copy

STEP = 6
MAX_W = 43
BLOCK_X_MAX = 48
BLOCK_Y_MIN, BLOCK_Y_MAX = 2, 58


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def tag_of(b):
    for t in ("todo", "next", "collected"):
        if t in b["tags"]:
            return t
    return None


def set_tag(b, t):
    b["tags"] = [x for x in b["tags"] if x not in ("todo", "next", "collected")] + [t]


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); any other block overlapped by a solid rect or moved block joins.
    Returns new positions dict or None if any moved block leaves bounds or enters the ring."""
    moving = set(movers)
    while True:
        rects = list(solids) + [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, 4, 4) for i in moving]
        added = False
        for i, b in enumerate(blocks):
            if i not in moving and any(overlap(r, rect(b)) for r in rects):
                moving.add(i)
                added = True
        if not added:
            break
    return {i: (blocks[i]["x"] + dx, blocks[i]["y"] + dy) for i in moving}


def valid(pos, ring):
    for x, y in pos.values():
        if not (BLOCK_Y_MIN <= y <= BLOCK_Y_MAX) or x > BLOCK_X_MAX:
            return False
        if overlap((x, y, 4, 4), rect(ring)):
            return False
    return True


def apply(blocks, pos):
    for i, (x, y) in pos.items():
        blocks[i]["x"], blocks[i]["y"] = x, y


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 5
    for i, b in enumerate(blocks):
        if b["x"] == tip and on_beam(b, beam) and tag_of(b) in ("next", "collected"):
            return i
    return None


def move_vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0 or ny + ring["h"] > 64:
        return
    carried = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
    new_ring = (ring["x"], ny, ring["w"], ring["h"])
    new_beam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    pos = push_chain(blocks, carried, [new_ring, new_beam], 0, dy)
    if not valid(pos, {"x": ring["x"], "y": ny, "w": ring["w"], "h": ring["h"]}):
        return
    apply(blocks, pos)
    ring["y"] = ny
    beam["y"] += dy


def move_horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return
    h = hooked(blocks, beam)
    if h is not None:
        group = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        pos = push_chain(blocks, group, [], dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        hit = [i for i, b in enumerate(blocks) if overlap(swept, rect(b))]
        pos = push_chain(blocks, hit, [], dw, 0) if hit else {}
    else:
        pos = {}
    if pos and valid(pos, ring):
        apply(blocks, pos)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)


def update_tags(beam, blocks):
    for b in blocks:
        if tag_of(b) == "collected" and not on_beam(b, beam):
            for o in blocks:
                if tag_of(o) == "next":
                    set_tag(o, "todo")
            set_tag(b, "next")
            return
    h = hooked(blocks, beam)
    if h is None or tag_of(blocks[h]) != "next":
        return
    others = [b for i, b in enumerate(blocks) if i != h and on_beam(b, beam)]
    if any(tag_of(b) != "collected" for b in others):
        return
    set_tag(blocks[h], "collected")
    todos = [b for b in blocks if tag_of(b) == "todo"]
    if todos:
        set_tag(max(todos, key=lambda b: b["x"]), "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o.get("type") == "player"), None)
    beam = next((o for o in state if o.get("type") == "arm"), None)
    blocks = [o for o in state if o.get("type") == "block"]
    if ring is None or beam is None or isinstance(action, dict):
        return state
    if action == 1:
        move_vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        move_vertical(ring, beam, blocks, STEP)
    elif action == 4:
        move_horizontal(ring, beam, blocks, STEP)
    elif action == 3:
        move_horizontal(ring, beam, blocks, -STEP)
    else:
        return state
    update_tags(beam, blocks)
    return state
