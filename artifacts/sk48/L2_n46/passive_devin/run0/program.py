# Mechanics: ring (player) + dashed beam (arm, h=2, w=1+6k, max 43) + 4x4 blocks tagged todo/next/collected.
# A1/A2: ring+beam y-/+6 carrying blocks on the beam; blocks hit get pushed (chain); any block y<2|y>58 cancels all.
# A4: beam w+6 (no-op at max w). Hooked block (next/collected at x == beam.x+w-5) drags on-beam blocks; else swept
# blocks are pushed; blocked group (x>48 or into ring) -> beam changes alone. A3: w-6, hooked group follows.
# Tags: hooked next alone on beam -> collected, rightmost todo -> next; unhooked collected reverts. Clicks/A5/A7 no-op.
import copy

STEP = 6
MAX_W = 43
BLOCK_MIN_Y, BLOCK_MAX_Y, BLOCK_MAX_X = 2, 58, 48


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return None


def set_status(b, s):
    b["tags"] = [s if t in ("collected", "next", "todo") else t for t in b["tags"]]


def push_chain(blocks, movers, solids, dx, dy):
    """Return {name: (x, y)} for moved blocks, or None if any block leaves bounds / enters the ring."""
    moving = {b["name"] for b in movers}
    changed = True
    while changed:
        changed = False
        moved_rects = [(b["x"] + dx, b["y"] + dy, b["w"], b["h"]) for b in blocks if b["name"] in moving]
        for b in blocks:
            if b["name"] in moving:
                continue
            if any(overlap(rect(b), r) for r in solids + moved_rects):
                moving.add(b["name"])
                changed = True
    return {b["name"]: (b["x"] + dx, b["y"] + dy) for b in blocks if b["name"] in moving}


def valid(blocks, moves, ring):
    for b in blocks:
        if b["name"] in moves:
            x, y = moves[b["name"]]
            if y < BLOCK_MIN_Y or y > BLOCK_MAX_Y or x > BLOCK_MAX_X or x < 0:
                return False
            if overlap((x, y, b["w"], b["h"]), rect(ring)):
                return False
    return True


def hooked_block(blocks, beam):
    tip = beam["x"] + beam["w"] - 5
    for b in blocks:
        if status(b) in ("next", "collected") and b["x"] == tip and overlap(rect(b), rect(beam)):
            return b
    return None


def on_beam(blocks, beam):
    return [b for b in blocks if overlap(rect(b), rect(beam))]


def apply(blocks, moves):
    for b in blocks:
        if b["name"] in moves:
            b["x"], b["y"] = moves[b["name"]]


def vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0 or ny + ring["h"] > 64:
        return
    carried = on_beam(blocks, beam)
    solids = [(ring["x"], ny, ring["w"], ring["h"]), (beam["x"], beam["y"] + dy, beam["w"], beam["h"])]
    moves = push_chain(blocks, carried, solids, 0, dy)
    if not valid(blocks, moves, {"x": ring["x"], "y": ny, "w": ring["w"], "h": ring["h"]}):
        return
    apply(blocks, moves)
    ring["y"] = ny
    beam["y"] += dy


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return
    hook = hooked_block(blocks, beam)
    if hook is not None:
        moves = push_chain(blocks, on_beam(blocks, beam), [], dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        moves = push_chain(blocks, [], [swept], dw, 0)
    else:
        moves = {}
    if valid(blocks, moves, ring):
        apply(blocks, moves)
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)


def update_tags(blocks, beam):
    hook = hooked_block(blocks, beam)
    if hook is not None and status(hook) == "next" and len(on_beam(blocks, beam)) == 1:
        set_status(hook, "collected")
        todos = [b for b in blocks if status(b) == "todo"]
        if todos:
            set_status(max(todos, key=lambda b: b["x"]), "next")
    for b in blocks:
        if status(b) == "collected" and b is not hook:
            for n in blocks:
                if status(n) == "next":
                    set_status(n, "todo")
            set_status(b, "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return state
    if action == 1:
        vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        vertical(ring, beam, blocks, STEP)
    elif action == 4:
        horizontal(ring, beam, blocks, STEP)
    elif action == 3:
        horizontal(ring, beam, blocks, -STEP)
    else:
        return state
    update_tags(blocks, beam)
    return state
