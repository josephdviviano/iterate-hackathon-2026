# Mechanics: crane game. ring (player) + beam (arm, h=2, dashed pixels, w=1+6k<=43) + 4x4 blocks.
# A1/A2: ring+beam y-/+6, blocks overlapping the beam are carried, blocks hit get pushed (chain);
#   any block leaving y in [2,58] cancels the whole move. A4/A3: beam w+/-6. Hooked (next/collected
#   block at x == tip-4) => on-beam blocks follow the tip; else extend pushes swept blocks (chain);
#   blocked group (x>48 or into ring) -> beam changes alone. Tags: hooked next alone on beam -> collected.
# Hypothesis "block A4 mixed outcome = global counter" rejected: it is the push chain hitting x>48. Ring y bounds guessed.
import copy

BLOCK_MIN_Y, BLOCK_MAX_Y, BLOCK_MAX_X, MAX_W = 2, 58, 48, 43


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    return b["tags"][-1]


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); any block hit by a solid rect or moved block is pushed too."""
    moved = {id(b): (b["x"] + dx, b["y"] + dy) for b in movers}
    changed = True
    while changed:
        changed = False
        rects = list(solids) + [(nx, ny, 4, 4) for nx, ny in moved.values()]
        for b in blocks:
            if id(b) in moved:
                continue
            if any(overlap(rect(b), r) for r in rects):
                moved[id(b)] = (b["x"] + dx, b["y"] + dy)
                changed = True
    return moved


def hooked_block(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    for b in blocks:
        if status(b) in ("next", "collected") and b["x"] == tip - 4 and overlap(rect(b), rect(beam)):
            return b
    return None


def on_beam(blocks, beam):
    return [b for b in blocks if overlap(rect(b), rect(beam))]


def vertical(ring, beam, blocks, dy):
    if ring["y"] + dy < 0 or ring["y"] + dy + ring["h"] > 64:
        return False
    carried = on_beam(blocks, beam)
    new_ring = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
    new_beam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    moved = push_chain(blocks, carried, [new_ring, new_beam], 0, dy)
    if any(not BLOCK_MIN_Y <= ny <= BLOCK_MAX_Y for _, ny in moved.values()):
        return False
    ring["y"] += dy
    beam["y"] += dy
    apply(blocks, moved)
    return True


def apply(blocks, moved):
    for b in blocks:
        if id(b) in moved:
            b["x"], b["y"] = moved[id(b)]


def group_ok(moved, ring):
    return all(ring["x"] + ring["w"] <= nx <= BLOCK_MAX_X for nx, _ in moved.values())


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return
    hook = hooked_block(blocks, beam)
    if hook is not None:
        moved = push_chain(blocks, on_beam(blocks, beam), [], dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        moved = push_chain(blocks, [], [swept], dw, 0)
    else:
        moved = {}
    if group_ok(moved, ring):
        apply(blocks, moved)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)


def update_tags(blocks, beam):
    hook = hooked_block(blocks, beam)
    todos = sorted([b for b in blocks if status(b) == "todo"], key=lambda b: -b["x"])
    if hook is not None and status(hook) == "next" and on_beam(blocks, beam) == [hook]:
        set_status(hook, "collected")
        if todos:
            set_status(todos[0], "next")
    for b in blocks:
        if status(b) == "collected" and b is not hook:
            set_status(b, "next")
            others = [o for o in blocks if o is not b and status(o) == "next"]
            for o in others:
                set_status(o, "todo")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return state
    if action == 1:
        vertical(ring, beam, blocks, -6)
    elif action == 2:
        vertical(ring, beam, blocks, 6)
    elif action == 4:
        horizontal(ring, beam, blocks, 6)
    elif action == 3:
        horizontal(ring, beam, blocks, -6)
    else:
        return state
    update_tags(blocks, beam)
    return state
