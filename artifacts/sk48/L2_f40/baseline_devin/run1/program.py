# Mechanics: ring (player) + beam (arm) form a crane; ACTION1/2 move crane y-/+6, ACTION4/3 extend/retract beam by 6 (w in 1..43).
# Blocks overlapped by the beam are skewered and travel with it; blocks newly hit by the beam or by moving blocks are pushed 6 (chains).
# A vertical move that would push anything out of bounds (y>=2, y+h<=62) or into the ring is cancelled entirely; a horizontal one
# lets the beam slide through the blocks instead (blocks stay). Clicks / other actions are no-ops. Tags: skewered blocks ordered
# outward from the ring that match the collection order are 'collected', the following one 'next'. Unconfirmed: exact bounds, todo order.
import copy

STEP, WMAX, YMIN, YMAX, XMAX = 6, 43, 2, 62, 53


def rect(o):
    return (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy)


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    return b["tags"][-1]


def try_move(blocks, carried, new_beam, ring_rect, dx, dy):
    """Return set of moving block indices, or None if blocked."""
    rects = [rect(b) for b in blocks]
    moving = set(carried)
    frontier = [i for i in range(len(blocks)) if i not in moving and overlap(rects[i], new_beam)]
    moving |= set(frontier)
    frontier = list(moving)
    while frontier:
        nxt = []
        for i in frontier:
            ri = shift(rects[i], dx, dy)
            for j in range(len(blocks)):
                if j not in moving and overlap(ri, rects[j]):
                    moving.add(j)
                    nxt.append(j)
        frontier = nxt
    for i in moving:
        r = shift(rects[i], dx, dy)
        if r[1] < YMIN or r[3] > YMAX or r[2] > XMAX or overlap(r, ring_rect):
            return None
    return moving


def collection_order(blocks):
    done = sorted([b for b in blocks if status(b) == "collected"], key=lambda b: b["x"])
    nxt = [b for b in blocks if status(b) == "next"]
    todo = sorted([b for b in blocks if status(b) == "todo"], key=lambda b: -b["x"])
    return [b["name"] for b in done + nxt + todo]


def retag(blocks, beam, order):
    br = rect(beam)
    skewered = [b["name"] for b in sorted(blocks, key=lambda b: b["x"]) if overlap(rect(b), br)]
    k = 0
    while k < len(skewered) and k < len(order) and skewered[k] == order[k]:
        k += 1
    for b in blocks:
        idx = order.index(b["name"]) if b["name"] in order else len(order)
        st = "collected" if idx < k else ("next" if idx == k else "todo")
        b["tags"] = b["tags"][:-1] + [st]


def transition_function(state, action):
    s = copy.deepcopy(state)
    ring = next((o for o in s if o.get("type") == "player"), None)
    beam = next((o for o in s if o.get("type") == "arm"), None)
    blocks = [o for o in s if o.get("type") == "block"]
    if ring is None or beam is None or isinstance(action, dict) or action not in (1, 2, 3, 4):
        return s
    order = collection_order(blocks)
    br = rect(beam)
    carried = [i for i, b in enumerate(blocks) if overlap(rect(b), br)]
    if action in (1, 2):
        dy = -STEP if action == 1 else STEP
        new_ring = shift(rect(ring), 0, dy)
        new_beam = shift(br, 0, dy)
        if new_ring[1] < YMIN or new_ring[3] > YMAX:
            return s
        moving = try_move(blocks, carried, new_beam, new_ring, 0, dy)
        if moving is None:
            return s
        ring["y"] += dy
        beam["y"] += dy
        for i in moving:
            blocks[i]["y"] += dy
    else:
        dw = STEP if action == 4 else -STEP
        w = beam["w"] + dw
        if w < 1 or w > WMAX or beam["x"] + w > XMAX:
            return s
        new_beam = (br[0], br[1], br[0] + w, br[3])
        moving = try_move(blocks, carried, new_beam, rect(ring), dw, 0)
        beam["w"] = w
        beam["pixels"] = beam_pixels(w)
        for i in moving or ():
            blocks[i]["x"] += dw
    retag(blocks, beam, order)
    return s
