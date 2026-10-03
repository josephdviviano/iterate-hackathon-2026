# Mechanics: crane ring (player) + beam (arm, h=2, w=1+6k<=43) + 4x4 blocks tagged todo/next/collected.
# A1/A2: ring+beam y-/+6, blocks overlapping beam are carried, blocks hit get pushed (chain); cancel if a block leaves y[2,58].
# A4/A3: beam w+/-6. Hooked (next/collected block at x==tip-4 on beam) -> all on-beam blocks move +/-6; unhooked extend
# pushes next/collected blocks in the swept cols (chain); if any moved block leaves x<=48 / enters ring, beam changes alone.
# Tags: hooked next with all other on-beam blocks collected -> collected, rightmost todo -> next; no hook & collected off beam -> reverts.
# Stateless: no hidden counter/turn order was needed. Ring y bounds [0,58] assumed (unobserved). Clicks (A6) and A5/A7 are no-ops.
import copy

STEP = 6
MAX_W = 43
MAX_X = 48
MIN_Y, MAX_Y = 2, 58


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def tag(b):
    return b["tags"][-1]


def set_tag(b, t):
    b["tags"] = b["tags"][:-1] + [t]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)],
            [2 if i % 3 == 0 else 1 for i in range(w)]]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 5
    for b in blocks:
        if tag(b) in ("next", "collected") and b["x"] == tip and on_beam(b, beam):
            return b
    return None


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); any other block hit by a mover's new rect or a solid rect is pushed too."""
    moved = set(id(m) for m in movers)
    queue = list(movers)
    for m in movers:
        m["x"] += dx
        m["y"] += dy
    hitters = [rect(m) for m in movers] + list(solids)
    while hitters:
        r = hitters.pop()
        for b in blocks:
            if id(b) not in moved and overlap(r, rect(b)):
                moved.add(id(b))
                b["x"] += dx
                b["y"] += dy
                hitters.append(rect(b))
                queue.append(b)
    return queue


def block_ok(b, ring):
    return (MIN_Y <= b["y"] <= MAX_Y and b["x"] <= MAX_X
            and b["x"] >= ring["x"] + ring["w"])


def vertical(ring, beam, blocks, dy):
    if not (0 <= ring["y"] + dy <= MAX_Y):
        return False
    carried = [b for b in blocks if on_beam(b, beam)]
    ring["y"] += dy
    beam["y"] += dy
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(b, ring) for b in moved)


def horizontal(ring, beam, blocks, dw):
    """Returns (beam_changed, blocks_valid)."""
    nw = beam["w"] + dw
    if nw < 1 or nw > MAX_W:
        return False, False
    hook = hooked(blocks, beam)
    if hook is not None:
        group = [b for b in blocks if on_beam(b, beam)]
        moved = push_chain(blocks, group, [], dw, 0)
    elif dw > 0:
        swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
        movers = [b for b in blocks if tag(b) in ("next", "collected") and overlap(swept, rect(b))]
        moved = push_chain(blocks, movers, [], dw, 0) if movers else []
    else:
        moved = []
    beam["w"] = nw
    return True, all(block_ok(b, ring) for b in moved)


def update_tags(blocks, beam):
    hook = hooked(blocks, beam)
    if hook is None:
        off = [b for b in blocks if tag(b) == "collected" and not on_beam(b, beam)]
        if off:
            top = max(off, key=lambda b: b["x"])
            for b in blocks:
                if tag(b) == "next":
                    set_tag(b, "todo")
            set_tag(top, "next")
        return
    if tag(hook) != "next":
        return
    others = [b for b in blocks if b is not hook and on_beam(b, beam)]
    if all(tag(b) == "collected" for b in others):
        set_tag(hook, "collected")
        todo = [b for b in blocks if tag(b) == "todo"]
        if todo:
            set_tag(max(todo, key=lambda b: (b["x"], -b["y"])), "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    blocks = [o for o in state if o["type"] == "block"]
    aid = action["action_id"] if isinstance(action, dict) else action
    if ring is None or beam is None or aid not in (1, 2, 3, 4):
        return state

    trial = copy.deepcopy(state)
    t_ring = next(o for o in trial if o["type"] == "player")
    t_beam = next(o for o in trial if o["type"] == "arm")
    t_blocks = [o for o in trial if o["type"] == "block"]

    if aid in (1, 2):
        if not vertical(t_ring, t_beam, t_blocks, -STEP if aid == 1 else STEP):
            return state
        result, r_beam, r_blocks = trial, t_beam, t_blocks
    else:
        changed, ok = horizontal(t_ring, t_beam, t_blocks, STEP if aid == 4 else -STEP)
        if not changed:
            return state
        if ok:
            result, r_beam, r_blocks = trial, t_beam, t_blocks
        else:
            beam["w"] = t_beam["w"]
            result, r_beam, r_blocks = state, beam, blocks
    r_beam["pixels"] = beam_pixels(r_beam["w"])
    update_tags(r_blocks, r_beam)
    return result
