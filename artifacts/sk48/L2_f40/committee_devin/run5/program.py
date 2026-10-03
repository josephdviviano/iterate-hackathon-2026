# Crane/skewer game: ring (player) at fixed x moves vertically; beam (arm) at ring.x+5, ring.y+2, h=2, w=1+6k (1..43).
# A1/A2: ring+beam y -/+6; blocks overlapping the beam are carried, blocks hit by moved rects are pushed (chain);
# any block leaving y in [2,58] or ring out of bounds cancels the whole move. A4/A3: beam w +/-6. If the next/collected
# block is hooked at the tip (x == beam end-5) all on-beam blocks follow the tip, else extension pushes swept blocks (chain);
# a blocked group (x>48 or into ring) lets the beam change alone. Tags: hooked next alone on beam -> collected (unconfirmed: order).
import copy

STEP = 6
MAX_W = 43
BLOCK_MAX_X = 48
BLOCK_MIN_Y, BLOCK_MAX_Y = 2, 58


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)],
            [2 if i % 3 == 0 else 1 for i in range(w)]]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def status(b):
    for t in ("todo", "next", "collected"):
        if t in b["tags"]:
            return t
    return None


def set_status(b, s):
    b["tags"] = [s if t in ("todo", "next", "collected") else t for t in b["tags"]]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def is_hooked(b, beam):
    return (status(b) in ("next", "collected") and b["x"] == beam["x"] + beam["w"] - 5
            and b["y"] < beam["y"] + beam["h"] and beam["y"] < b["y"] + b["h"])


def push_chain(blocks, moved, dx, dy, obstacles):
    """moved: set of indices already displaced; push others hit by obstacles/moved blocks."""
    changed = True
    while changed:
        changed = False
        rects = list(obstacles) + [rect(blocks[i]) for i in moved]
        for i, b in enumerate(blocks):
            if i in moved:
                continue
            if any(overlap(rect(b), r) for r in rects):
                b["x"] += dx
                b["y"] += dy
                moved.add(i)
                changed = True
                break
    return moved


def blocks_ok(blocks, ring):
    ring_r = rect(ring)
    for b in blocks:
        if b["x"] > BLOCK_MAX_X or b["y"] < BLOCK_MIN_Y or b["y"] > BLOCK_MAX_Y:
            return False
        if overlap(rect(b), ring_r):
            return False
    return True


def vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0 or ny + ring["h"] > 64:
        return False
    nb = copy.deepcopy(blocks)
    moved = set()
    for i, b in enumerate(nb):
        if on_beam(blocks[i], beam):
            b["y"] += dy
            moved.add(i)
    nring = dict(ring, y=ny)
    nbeam = dict(beam, y=beam["y"] + dy)
    push_chain(nb, moved, 0, dy, [rect(nring), rect(nbeam)])
    if not blocks_ok(nb, nring):
        return False
    ring["y"], beam["y"] = nring["y"], nbeam["y"]
    for b, n in zip(blocks, nb):
        b["x"], b["y"] = n["x"], n["y"]
    return True


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return False
    hooked = any(is_hooked(b, beam) for b in blocks)
    nb = copy.deepcopy(blocks)
    nbeam = dict(beam, w=nw)
    if hooked:
        for i, b in enumerate(nb):
            if on_beam(blocks[i], beam):
                b["x"] += dw
    elif dw > 0:
        old_end = beam["x"] + beam["w"]
        swept = (old_end, beam["y"], dw, beam["h"])
        moved = set()
        for i, b in enumerate(nb):
            if overlap(rect(b), swept):
                b["x"] += dw
                moved.add(i)
        push_chain(nb, moved, dw, 0, [])
    if blocks_ok(nb, ring):
        for b, n in zip(blocks, nb):
            b["x"], b["y"] = n["x"], n["y"]
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    return True


def update_tags(beam, blocks):
    for b in blocks:
        if status(b) == "collected" and not is_hooked(b, beam):
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
            return
    for b in blocks:
        if status(b) == "next" and is_hooked(b, beam):
            if all(o is b or not on_beam(o, beam) for o in blocks):
                set_status(b, "collected")
                todo = [o for o in blocks if status(o) == "todo"]
                if todo:
                    set_status(max(todo, key=lambda o: o["x"]), "next")
            return


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return state
    if action in (1, 2):
        changed = vertical(ring, beam, blocks, -STEP if action == 1 else STEP)
    elif action in (3, 4):
        changed = horizontal(ring, beam, blocks, -STEP if action == 3 else STEP)
    else:
        changed = False
    if changed:
        update_tags(beam, blocks)
    return state
