# Mechanics: crane/skewer game. A1/A2 move ring+beam y-/+6, carrying on-beam blocks and pushing hit blocks (chain); cancel if a block leaves y [2,58].
# A4/A3 change beam w by 6 (1..43, dashed pixels). Hooked = next/collected block on beam rows at x == tip-5 -> on-beam blocks follow +-6;
# unhooked extend pushes swept blocks (chain); a blocked group (x>48 or into ring) stays and the beam moves alone. A5-A7 no-ops.
# Tags: hooked next with only collected others on beam -> collected, rightmost todo -> next; nothing hooked -> top collected -> next, next -> todo.
# Unconfirmed: ring's own y bounds, reverting several collected blocks, todo tie-breaks. Stateless (no hidden state).
import copy

STEP = 6
MAX_W = 43
MAX_BX = 48
MIN_BY, MAX_BY = 2, 58


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def tags_of(b):
    return b["tags"][-1]


def set_tag(b, t):
    b["tags"] = b["tags"][:-1] + [t]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def rows_overlap(b, beam):
    return b["y"] < beam["y"] + beam["h"] and beam["y"] < b["y"] + b["h"]


def hooked_block(blocks, beam):
    tip = beam["x"] + beam["w"]
    for b in blocks:
        if tags_of(b) in ("next", "collected") and b["x"] == tip - 5 and rows_overlap(b, beam):
            return b
    return None


def push_chain(blocks, movers, solids, dx, dy):
    """movers: set of block indices moving by (dx,dy); solids: rects (already moved) that push blocks.
    Returns dict idx->(x,y) of all moved blocks."""
    moved = set(movers)
    changed = True
    while changed:
        changed = False
        pushers = list(solids) + [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, blocks[i]["w"], blocks[i]["h"]) for i in moved]
        for j, b in enumerate(blocks):
            if j in moved:
                continue
            if any(overlap(rect(b), p) for p in pushers):
                moved.add(j)
                changed = True
    return {i: (blocks[i]["x"] + dx, blocks[i]["y"] + dy) for i in moved}


def valid_block_pos(b, x, y, ring):
    if y < MIN_BY or y > MAX_BY or x > MAX_BX:
        return False
    return not overlap((x, y, b["w"], b["h"]), rect(ring))


def apply_moves(blocks, moves, ring):
    if all(valid_block_pos(blocks[i], x, y, ring) for i, (x, y) in moves.items()):
        for i, (x, y) in moves.items():
            blocks[i]["x"], blocks[i]["y"] = x, y
        return True
    return False


def vertical(ring, beam, blocks, dy):
    carried = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
    new_ring = (ring["x"], ring["y"] + dy, ring["w"], ring["h"])
    new_beam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
    if new_ring[1] < 0 or new_ring[1] + new_ring[3] > 64:
        return
    moves = push_chain(blocks, carried, [new_ring, new_beam], 0, dy)
    if not all(valid_block_pos(blocks[i], x, y, {"x": new_ring[0], "y": new_ring[1], "w": new_ring[2], "h": new_ring[3]})
               for i, (x, y) in moves.items()):
        return
    for i, (x, y) in moves.items():
        blocks[i]["x"], blocks[i]["y"] = x, y
    ring["y"] += dy
    beam["y"] += dy


def horizontal(ring, beam, blocks, dw):
    new_w = beam["w"] + dw
    if new_w < 1 or new_w > MAX_W:
        return
    hook = hooked_block(blocks, beam)
    if hook is not None:
        group = [i for i, b in enumerate(blocks) if on_beam(b, beam)]
        moves = push_chain(blocks, group, [], dw, 0)
        apply_moves(blocks, moves, ring)
    elif dw > 0:
        tip = beam["x"] + beam["w"]
        swept = (tip, beam["y"], dw, beam["h"])
        moves = push_chain(blocks, [], [swept], dw, 0)
        if moves:
            apply_moves(blocks, moves, ring)
    beam["w"] = new_w
    beam["pixels"] = beam_pixels(new_w)


def update_tags(beam, blocks):
    hook = hooked_block(blocks, beam)
    collected = sorted([b for b in blocks if tags_of(b) == "collected"], key=lambda b: -b["x"])
    nxt = [b for b in blocks if tags_of(b) == "next"]
    if collected and hook is None:
        for b in nxt:
            set_tag(b, "todo")
        set_tag(collected[0], "next")
        return
    if hook is not None and tags_of(hook) == "next":
        others = [b for b in blocks if b is not hook and on_beam(b, beam)]
        if all(tags_of(b) == "collected" for b in others):
            set_tag(hook, "collected")
            todos = sorted([b for b in blocks if tags_of(b) == "todo"], key=lambda b: (-b["x"], b["y"]))
            if todos:
                set_tag(todos[0], "next")


def transition_function(state, action):
    st = copy.deepcopy(state)
    ring = next((o for o in st if o["type"] == "player"), None)
    beam = next((o for o in st if o["type"] == "arm"), None)
    blocks = [o for o in st if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int):
        return st
    if action == 1:
        vertical(ring, beam, blocks, -STEP)
    elif action == 2:
        vertical(ring, beam, blocks, STEP)
    elif action == 4:
        horizontal(ring, beam, blocks, STEP)
    elif action == 3:
        horizontal(ring, beam, blocks, -STEP)
    else:
        return st
    update_tags(beam, blocks)
    return st
