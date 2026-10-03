# Mechanics: crane game. ring (player) + beam (arm) move together vertically by 6 (A1 up, A2 down), carrying
# blocks overlapping the beam; blocks hit by the new ring/beam rects are pushed in a chain; cancel if any block y<2.
# A4/A3 extend/retract the beam by 6 (w in [1,43]). Hooked = next/collected block at x == tip-4 on the beam: all
# on-beam blocks follow; unhooked extend pushes swept blocks; blocked group (x>48 or into ring) -> beam moves alone.
# Tags: hooked next alone on beam -> collected (rightmost todo -> next); unhooked collected reverts. No hidden state needed.
import copy

STEP, WMAX, YMIN, XMAX = 6, 43, 2, 48


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
    """Move `movers` by (dx,dy); any block overlapping a solid or a moved block is pushed too."""
    moved = set(movers)
    frontier = [rect(blocks[i]) for i in movers] + list(solids)
    frontier = [(r[0] + dx, r[1] + dy, r[2], r[3]) if k < len(movers) else r for k, r in enumerate(frontier)]
    while frontier:
        r = frontier.pop()
        for i, b in enumerate(blocks):
            if i not in moved and overlap(rect(b), r):
                moved.add(i)
                frontier.append((b["x"] + dx, b["y"] + dy, b["w"], b["h"]))
    for i in moved:
        blocks[i]["x"] += dx
        blocks[i]["y"] += dy
    return moved


def block_ok(b, ring):
    return YMIN <= b["y"] <= 58 and b["x"] <= XMAX and b["x"] >= ring["x"] + ring["w"]


def on_beam(blocks, beam):
    return [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    for i in on_beam(blocks, beam):
        if status(blocks[i]) in ("next", "collected") and blocks[i]["x"] == tip - 4:
            return i
    return None


def vertical(ring, beam, blocks, dy):
    carried = on_beam(blocks, beam)
    ring["y"] += dy
    beam["y"] += dy
    if ring["y"] < 0 or ring["y"] + ring["h"] > 64:
        return False
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(blocks[i], ring) for i in moved)


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or nw > WMAX:
        return False
    trial = copy.deepcopy(blocks)
    h = hooked(blocks, beam)
    old_tip = beam["x"] + beam["w"] - 1
    if h is not None:
        moved = push_chain(trial, on_beam(blocks, beam), [], dw, 0)
    elif dw > 0:
        swept = (old_tip + 1, beam["y"], dw, beam["h"])
        moved = push_chain(trial, [], [swept], dw, 0)
    else:
        moved = set()
    beam["w"] = nw
    if all(block_ok(trial[i], ring) for i in moved):
        blocks[:] = trial
    return True


def update_tags(blocks, beam):
    h = hooked(blocks, beam)
    for b in blocks:
        if status(b) == "collected" and (h is None or blocks[h] is not b):
            nxt = [c for c in blocks if status(c) == "next"]
            for c in nxt:
                set_status(c, "todo")
            set_status(b, "next")
    h = hooked(blocks, beam)
    if h is not None and status(blocks[h]) == "next" and on_beam(blocks, beam) == [h]:
        set_status(blocks[h], "collected")
        todo = [c for c in blocks if status(c) == "todo"]
        if todo:
            set_status(max(todo, key=lambda c: c["x"]), "next")


def transition_function(state, action):
    state = copy.deepcopy(state)
    ring = next((o for o in state if o["type"] == "player"), None)
    beam = next((o for o in state if o["type"] == "arm"), None)
    if ring is None or beam is None or not isinstance(action, int):
        return state
    blocks = [o for o in state if o["type"] == "block"]
    others = [o for o in state if o["type"] not in ("player", "arm", "block")]
    r2, b2, bl2 = copy.deepcopy(ring), copy.deepcopy(beam), copy.deepcopy(blocks)
    if action in (1, 2):
        ok = vertical(r2, b2, bl2, -STEP if action == 1 else STEP)
    elif action in (3, 4):
        ok = horizontal(r2, b2, bl2, STEP if action == 4 else -STEP)
    else:
        ok = False
    if not ok:
        return state
    b2["pixels"] = beam_pixels(b2["w"])
    update_tags(bl2, b2)
    return [r2, b2] + bl2 + others
