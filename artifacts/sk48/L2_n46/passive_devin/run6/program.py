# Mechanics: ring (player) + beam (arm, at ring.x+5,ring.y+2, h=2, dashed pixels) + 4x4 blocks tagged todo/next/collected.
# A1/A2: ring+beam y-/+6, blocks on the beam are carried, blocks hit are pushed (chain); any block y outside [2,58] cancels.
# A4/A3: beam w+/-6 in [1,43]. Hooked (next/collected block at x==beam.x+w-5) -> on-beam blocks follow the tip;
#   unhooked extend pushes swept blocks +6 (chain); a blocked group (x>48 or into ring) leaves the beam to move alone.
# Tags: unhooked collected -> next (old next -> todo); hooked next alone on beam -> collected, rightmost todo -> next. Unconfirmed: ring y bounds, clicks (no-op).
import copy

W_MAX, W_MIN, STEP, X_MAX, Y_MIN, Y_MAX = 43, 1, 6, 48, 2, 58


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def tag(o):
    return o["tags"][3] if len(o["tags"]) > 3 else None


def set_tag(o, t):
    o["tags"] = o["tags"][:3] + [t] + o["tags"][4:]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def is_hooked(b, beam):
    return tag(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == beam["x"] + beam["w"] - 5


def block_ok(b, ring):
    return Y_MIN <= b["y"] <= Y_MAX and b["x"] + b["w"] <= X_MAX + 4 and b["x"] >= ring["x"] + ring["w"]


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); any other block overlapping a solid rect or moved block is pushed too."""
    moved = {id(b) for b in movers}
    for b in movers:
        b["x"] += dx
        b["y"] += dy
    frontier = list(solids) + [rect(b) for b in movers]
    while frontier:
        r = frontier.pop()
        for b in blocks:
            if id(b) not in moved and overlap(rect(b), r):
                moved.add(id(b))
                b["x"] += dx
                b["y"] += dy
                frontier.append(rect(b))
    return [b for b in blocks if id(b) in moved]


def vertical(ring, beam, blocks, dy):
    if not (0 <= ring["y"] + dy <= Y_MAX):
        return False
    carried = [b for b in blocks if on_beam(b, beam)]
    ring["y"] += dy
    beam["y"] += dy
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(b, ring) for b in moved)


def horizontal(ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if not (W_MIN <= nw <= W_MAX):
        return None
    hooked = any(is_hooked(b, beam) for b in blocks)
    old_end = beam["x"] + beam["w"]
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw)
    if hooked:
        group = [b for b in blocks if overlap(rect(b), (beam["x"], beam["y"], max(nw, nw - dw), 2))]
        moved = push_chain(blocks, group, [], dw, 0)
    elif dw > 0:
        swept = (old_end, beam["y"], dw, beam["h"])
        moved = push_chain(blocks, [], [swept], dw, 0)
    else:
        moved = []
    return moved


def update_tags(blocks, beam):
    for b in blocks:
        if tag(b) == "collected" and not is_hooked(b, beam):
            for o in blocks:
                if tag(o) == "next":
                    set_tag(o, "todo")
            set_tag(b, "next")
    for b in blocks:
        if tag(b) == "next" and is_hooked(b, beam) and all(o is b or not on_beam(o, beam) for o in blocks):
            set_tag(b, "collected")
            todos = [o for o in blocks if tag(o) == "todo"]
            if todos:
                set_tag(max(todos, key=lambda o: o["x"]), "next")


def step(state, action):
    s = copy.deepcopy(state)
    ring = next((o for o in s if o["type"] == "player"), None)
    beam = next((o for o in s if o["type"] == "arm"), None)
    blocks = [o for o in s if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int) or action not in (1, 2, 3, 4):
        return s
    if action in (1, 2):
        if not vertical(ring, beam, blocks, -STEP if action == 1 else STEP):
            return copy.deepcopy(state)
    else:
        dw = STEP if action == 4 else -STEP
        trial = copy.deepcopy(s)
        t_ring = next(o for o in trial if o["type"] == "player")
        t_beam = next(o for o in trial if o["type"] == "arm")
        t_blocks = [o for o in trial if o["type"] == "block"]
        moved = horizontal(t_ring, t_beam, t_blocks, dw)
        if moved is None:
            return copy.deepcopy(state)
        if all(block_ok(b, t_ring) for b in moved):
            s = trial
        else:
            beam["w"] += dw
            beam["pixels"] = beam_pixels(beam["w"])
        beam = next(o for o in s if o["type"] == "arm")
        blocks = [o for o in s if o["type"] == "block"]
    update_tags(blocks, beam)
    return s


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    return step(state, action)
