# Mechanics: ring (player) + dashed beam (arm, h=2, w=1+6k<=43) + 4x4 blocks tagged todo/next/collected.
# A1/A2: ring+beam y-/+6, blocks overlapping the beam are carried, blocks hit get pushed (chain);
#   any block leaving y in [2,58] cancels the whole move (this decides block A1 y vs no_change).
# A4/A3: beam w+/-6. Hooked (next/collected block at x == beam end-5) drags all on-beam blocks;
#   unhooked extend pushes swept blocks; blocked group (x>48 / into ring) -> beam alone. Clicks no-op.
import copy

BW, STEP, WMAX, YMIN, YMAX, XMAX = 4, 6, 43, 2, 58, 48


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


def push_chain(blocks, movers, solids, dx, dy):
    """Return set of indices of blocks moved by (dx,dy): movers plus anything hit transitively."""
    moved = set(movers)
    changed = True
    while changed:
        changed = False
        occ = list(solids) + [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, BW, BW) for i in moved]
        for i, b in enumerate(blocks):
            if i not in moved and any(overlap(rect(b), r) for r in occ):
                moved.add(i)
                changed = True
    return moved


def valid(blocks, moved, dx, dy, ring):
    for i in moved:
        x, y = blocks[i]["x"] + dx, blocks[i]["y"] + dy
        if y < YMIN or y > YMAX or x > XMAX or x < ring["x"] + ring["w"]:
            return False
    return True


def apply(blocks, moved, dx, dy):
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy


def beam_end(beam):
    return beam["x"] + beam["w"]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def is_hooked(b, beam):
    return (status(b) in ("next", "collected") and b["x"] == beam_end(beam) - 5
            and overlap((0, b["y"], 1, BW), (0, beam["y"], 1, beam["h"])))


def hooked_any(blocks, beam):
    return any(is_hooked(b, beam) for b in blocks)


def set_width(beam, w):
    beam["w"] = w
    beam["pixels"] = beam_pixels(w)


def move_vertical(ring, beam, blocks, dy):
    ny = ring["y"] + dy
    if ny < 0 or ny + ring["h"] > 64:
        return
    carried = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
    solids = [(ring["x"], ny, ring["w"], ring["h"]), (beam["x"], beam["y"] + dy, beam["w"], beam["h"])]
    moved = push_chain(blocks, carried, solids, 0, dy)
    if not valid(blocks, moved, 0, dy, ring):
        return
    apply(blocks, moved, 0, dy)
    ring["y"] += dy
    beam["y"] += dy


def extend(ring, beam, blocks):
    if beam["w"] + STEP > WMAX:
        return
    if hooked_any(blocks, beam):
        movers = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        moved = push_chain(blocks, movers, [], STEP, 0)
    else:
        swept = (beam_end(beam), beam["y"], STEP, beam["h"])
        moved = push_chain(blocks, [], [swept], STEP, 0)
    if valid(blocks, moved, STEP, 0, ring):
        apply(blocks, moved, STEP, 0)
    set_width(beam, beam["w"] + STEP)


def retract(ring, beam, blocks):
    if beam["w"] - STEP < 1:
        return
    if hooked_any(blocks, beam):
        movers = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        moved = push_chain(blocks, movers, [], -STEP, 0)
        if valid(blocks, moved, -STEP, 0, ring):
            apply(blocks, moved, -STEP, 0)
    set_width(beam, beam["w"] - STEP)


def update_tags(beam, blocks):
    for b in blocks:
        if status(b) == "collected" and not is_hooked(b, beam):
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
    nxt = [b for b in blocks if status(b) == "next"]
    for b in nxt:
        if is_hooked(b, beam) and all(o is b or not on_beam(o, beam) for o in blocks):
            set_status(b, "collected")
            todos = [o for o in blocks if status(o) == "todo"]
            if todos:
                set_status(max(todos, key=lambda o: o["x"]), "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return state
    if action in (1, 2):
        move_vertical(ring, beam, blocks, -STEP if action == 1 else STEP)
    elif action == 4:
        extend(ring, beam, blocks)
    elif action == 3:
        retract(ring, beam, blocks)
    else:
        return state
    update_tags(beam, blocks)
    return state
