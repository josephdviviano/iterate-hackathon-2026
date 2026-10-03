# Crane/skewer game: ring (player) carries a dashed beam (arm, h=2, w=1+6k<=43) at ring.x+5, ring.y+2.
# A1/A2 move ring+beam y-/+6, carrying blocks on the beam and pushing blocks hit (chain); any block
# leaving y[2,58] cancels the move. A4/A3 change beam w by +-6; if a next/collected block is hooked at
# the tip (x == tip-4) all on-beam blocks follow, else extension pushes swept blocks; a blocked group
# (x>48 or into the ring) leaves the blocks and moves only the beam. Tags: hooked next with every other
# on-beam block collected -> collected (rightmost todo -> next); unhooked + collected off beam -> revert.
# Unconfirmed: ring y bounds, which collected block reverts when several are off the beam. Clicks no-op.
import copy

MAX_W, STEP = 43, 6


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    return b["tags"][-1]


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); blocks hit by solids or moved blocks are pushed too."""
    moved = set(movers)
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(blocks):
            if i in moved:
                continue
            hit = any(overlap(rect(b), s) for s in solids) or any(
                overlap(rect(b), rect(blocks[j])) for j in moved)
            if hit:
                b["x"] += dx
                b["y"] += dy
                moved.add(i)
                changed = True
    return moved


def block_ok(b, ring):
    return 2 <= b["y"] <= 58 and b["x"] <= 48 and b["x"] >= ring["x"] + ring["w"]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked_block(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    for i, b in enumerate(blocks):
        if status(b) in ("next", "collected") and b["x"] == tip - 4 and on_beam(b, beam):
            return i
    return None


def set_status(b, s):
    b["tags"] = b["tags"][:-1] + [s]


def update_tags(blocks, beam):
    h = hooked_block(blocks, beam)
    if h is not None:
        hb = blocks[h]
        others = [b for i, b in enumerate(blocks) if i != h and on_beam(b, beam)]
        if status(hb) == "next" and all(status(b) == "collected" for b in others):
            set_status(hb, "collected")
            todos = [b for b in blocks if status(b) == "todo"]
            if todos:
                set_status(max(todos, key=lambda b: b["x"]), "next")
        return
    off = [b for b in blocks if status(b) == "collected" and not on_beam(b, beam)]
    if off:
        for b in blocks:
            if status(b) == "next":
                set_status(b, "todo")
        set_status(max(off, key=lambda b: b["x"]), "next")


def step_vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if not 0 <= ny <= 58:
        return False
    trial = copy.deepcopy(blocks)
    carried = [i for i, b in enumerate(trial) if on_beam(b, beam)]
    nring = (ring["x"], ny, ring["w"], ring["h"])
    nbeam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    moved = push_chain(trial, carried, [nring, nbeam], 0, dy)
    if not all(block_ok(trial[i], ring) for i in moved):
        return False
    ring["y"] = ny
    beam["y"] += dy
    blocks[:] = trial
    return True


def step_horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return False
    trial = copy.deepcopy(blocks)
    h = hooked_block(blocks, beam)
    if h is not None:
        movers = [i for i, b in enumerate(trial) if on_beam(b, beam)]
        moved = push_chain(trial, movers, [], dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        moved = push_chain(trial, [], [swept], dw, 0)
    else:
        moved = set()
    if all(block_ok(trial[i], ring) for i in moved):
        blocks[:] = trial
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    return True


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o.get("type") == "player"), None)
    beam = next((o for o in state if o.get("type") == "arm"), None)
    if ring is None or beam is None or not isinstance(action, int):
        return state
    blocks = [o for o in state if o.get("type") == "block"]
    rest = [o for o in state if o.get("type") not in ("block",)]
    if action in (1, 2):
        step_vertical(ring, beam, blocks, -STEP if action == 1 else STEP)
    elif action in (3, 4):
        step_horizontal(ring, beam, blocks, STEP if action == 4 else -STEP)
    else:
        return state
    update_tags(blocks, beam)
    return rest + blocks
