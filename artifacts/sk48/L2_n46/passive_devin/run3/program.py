# Crane/skewer game: ring (player) at fixed x with a dashed beam (arm) at ring.x+5, ring.y+2, h=2, w=1+6k (max 43).
# A1/A2 move ring+beam y-/+6, carrying blocks overlapping the beam and pushing hit blocks (chain); any block y<2 or >58 cancels.
# A4/A3 extend/retract the beam by 6. Hooked (next/collected block at x == tip-4 on the beam) drags all on-beam blocks;
# unhooked extend pushes swept blocks (chain); a blocked group (x>48, x<0, into ring) lets the beam change alone.
# Tags: hooked 'next' alone on beam -> collected, rightmost todo -> next; unhooked collected -> next (old next -> todo). Clicks no-op.
import copy

STEP = 6
MAX_W = 43
BX_MAX = 48
BY_MIN, BY_MAX = 2, 58


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    return b["tags"][-1]


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def push_chain(blocks, initial, solids, dx, dy):
    moved = set(initial)
    changed = True
    while changed:
        changed = False
        rects = list(solids) + [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, 4, 4) for i in moved]
        for i, b in enumerate(blocks):
            if i not in moved and any(overlap(rect(b), r) for r in rects):
                moved.add(i)
                changed = True
    return moved


def blocks_valid(blocks, moved, dx, dy, ring_r):
    for i in moved:
        b = blocks[i]
        nx, ny = b["x"] + dx, b["y"] + dy
        if not (0 <= nx <= BX_MAX and BY_MIN <= ny <= BY_MAX):
            return False
        if overlap((nx, ny, b["w"], b["h"]), ring_r):
            return False
    return True


def apply(blocks, moved, dx, dy):
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy


def on_beam(blocks, beam):
    return [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]


def hooked_index(blocks, beam):
    tip_x = beam["x"] + beam["w"] - 5
    for i in on_beam(blocks, beam):
        if blocks[i]["x"] == tip_x and status(blocks[i]) in ("next", "collected"):
            return i
    return None


def move_vertical(ring, beam, blocks, dy):
    carried = on_beam(blocks, beam)
    new_ring = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
    new_beam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    if new_ring[1] < 0 or new_ring[1] + new_ring[3] > 64:
        return
    moved = push_chain(blocks, carried, [new_ring, new_beam], 0, dy)
    if not blocks_valid(blocks, moved, 0, dy, (-100, -100, 0, 0)):
        return
    apply(blocks, moved, 0, dy)
    ring["y"] += dy
    beam["y"] += dy


def change_beam(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return
    ring_r = rect(ring)
    if hooked_index(blocks, beam) is not None:
        moved = push_chain(blocks, on_beam(blocks, beam), [], dw, 0)
        if blocks_valid(blocks, moved, dw, 0, ring_r):
            apply(blocks, moved, dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        moved = push_chain(blocks, [], [swept], dw, 0)
        if blocks_valid(blocks, moved, dw, 0, ring_r):
            apply(blocks, moved, dw, 0)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)


def update_tags(blocks, beam):
    h = hooked_index(blocks, beam)
    for i, b in enumerate(blocks):
        if status(b) == "collected" and i != h:
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
            return
    if h is not None and status(blocks[h]) == "next" and on_beam(blocks, beam) == [h]:
        set_status(blocks[h], "collected")
        todos = [b for b in blocks if status(b) == "todo"]
        if todos:
            set_status(max(todos, key=lambda b: b["x"]), "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return state
    if action == 1:
        move_vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        move_vertical(ring, beam, blocks, STEP)
    elif action == 4:
        change_beam(ring, beam, blocks, STEP)
    elif action == 3:
        change_beam(ring, beam, blocks, -STEP)
    else:
        return state
    update_tags(blocks, beam)
    return state
