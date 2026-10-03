# Crane/skewer: ring (player) + beam (arm, x=ring.x+5, y=ring.y+2, h=2, w=1+6k<=43) + 4x4 blocks.
# A1/A2: ring+beam y-/+6, on-beam blocks carried, blocks hit get pushed (chain); any block y outside [2,58] cancels all.
# A4/A3: beam w+/-6. Hooked (next/collected block at x==tip-5) -> all on-beam blocks follow; else extend pushes swept
# blocks; if moved blocks leave x<=48 / ring-free range, beam changes alone. Tags: off-beam collected reverts to next;
# hooked next with only collected others on beam -> collected, rightmost todo -> next. Clicks no-op. Tie order unconfirmed.
import copy

STEP = 6
MAX_W = 43
MAX_X = 48
Y_MIN, Y_MAX = 2, 58


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def status(b):
    return b["tags"][-1]


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def tip(beam):
    return beam["x"] + beam["w"]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def is_hooked(b, beam):
    return status(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == tip(beam) - 5


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` and every block pushed by them or by `solids` (rects) by (dx, dy)."""
    moved = set(movers)
    for i, b in enumerate(blocks):
        if i not in moved and any(overlap(rect(b), s) for s in solids):
            moved.add(i)
    frontier = list(moved)
    while frontier:
        new = []
        for i in frontier:
            b = blocks[i]
            r = (b["x"] + dx, b["y"] + dy, b["w"], b["h"])
            for j, c in enumerate(blocks):
                if j not in moved and overlap(r, rect(c)):
                    moved.add(j)
                    new.append(j)
        frontier = new
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
    return moved


def block_ok(b, ring):
    return Y_MIN <= b["y"] <= Y_MAX and b["x"] <= MAX_X and b["x"] >= ring["x"] + ring["w"]


def vertical(ring, beam, blocks, dy):
    if not (0 <= ring["y"] + dy <= Y_MAX):
        return False
    carried = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
    ring["y"] += dy
    beam["y"] += dy
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(blocks[i], ring) for i in moved)


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if not (1 <= nw <= MAX_W):
        return False
    orig = copy.deepcopy(blocks)
    hooked = any(is_hooked(b, beam) for b in blocks)
    if hooked:
        movers = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        moved = push_chain(blocks, movers, [], dw, 0)
    elif dw > 0:
        swept = (tip(beam), beam["y"], dw, beam["h"])
        moved = push_chain(blocks, [], [swept], dw, 0)
    else:
        moved = set()
    if not all(block_ok(blocks[i], ring) for i in moved):
        blocks[:] = orig
    beam["w"] = nw
    return True


def update_tags(beam, blocks):
    for b in blocks:
        if status(b) == "collected" and not on_beam(b, beam):
            for c in blocks:
                if status(c) == "next":
                    set_status(c, "todo")
            set_status(b, "next")
    for b in blocks:
        if status(b) == "next" and is_hooked(b, beam):
            others = [c for c in blocks if c is not b and on_beam(c, beam)]
            if all(status(c) == "collected" for c in others):
                set_status(b, "collected")
                todos = [c for c in blocks if status(c) == "todo"]
                if todos:
                    set_status(max(todos, key=lambda c: (c["x"], -c["y"])), "next")
            break


def transition_function(state, action):
    state = copy.deepcopy(state)
    if isinstance(action, dict) or action not in (1, 2, 3, 4):
        return state
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    if ring is None or beam is None:
        return state
    trial = copy.deepcopy(state)
    t_ring = next(o for o in trial if o["type"] == "player")
    t_beam = next(o for o in trial if o["type"] == "arm")
    t_blocks = [o for o in trial if o["type"] == "block"]
    if action in (1, 2):
        ok = vertical(t_ring, t_beam, t_blocks, -STEP if action == 1 else STEP)
    else:
        ok = horizontal(t_ring, t_beam, t_blocks, STEP if action == 4 else -STEP)
    if not ok:
        return state
    update_tags(t_beam, t_blocks)
    t_beam["pixels"] = beam_pixels(t_beam["w"])
    out = [o for o in trial if o["type"] != "block"] + t_blocks
    return out
