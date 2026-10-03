# Mechanics: crane ring (player) + beam (arm) move A1/A2 by 6 rows, carrying blocks on the beam and pushing
# blocks they hit (whole move cancels if a block leaves the colour-4 field); A4/A3 change beam width by 6:
# a hooked next/collected block (x == tip-4) drags all on-beam blocks, else extending pushes swept next/collected
# blocks; a blocked group leaves the beam sliding alone. Legend swatch centre 2x2 = 0 while collected. HUD bar
# row turns one 2->3 from the right every 3 non-click actions (hidden counter, continuity-gated). Clicks no-op.
import copy

STEP = 6
_last = {"frame": None, "n": 0}


def find_hud_row(frame):
    for y, row in enumerate(frame):
        if all(v not in (4, 5) for v in row):
            return y
    return len(frame)


def find_field(frame, hud):
    xs = [x for y in range(hud) for x in range(64) if frame[y][x] == 4]
    ys = [y for y in range(hud) for x in range(64) if frame[y][x] == 4]
    return min(xs), min(ys), max(xs), max(ys)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.hud = find_hud_row(frame)
        self.field = find_field(frame, self.hud)
        self.ring = next(o for o in state if o["type"] == "player")
        self.beam = next(o for o in state if o["type"] == "arm")
        self.blocks = [o for o in state if o["type"] == "block"]

    def block_ok(self, b):
        l, t, r, bt = self.field
        return b["x"] >= l and b["x"] + b["w"] - 1 <= r and b["y"] >= t and b["y"] + b["h"] - 1 <= bt

    def ring_ok(self, ring):
        l, t, r, bt = self.field
        return ring["y"] >= t and ring["y"] + ring["h"] - 1 <= bt


def tag(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def set_tag(b, t):
    b["tags"] = [x for x in b["tags"] if x not in ("collected", "next", "todo")] + [t]


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); any other block overlapped by a moved rect or solid gets pushed too."""
    moved = set()
    queue = list(movers)
    while queue:
        b = queue.pop()
        if id(b) in moved:
            continue
        b["x"] += dx
        b["y"] += dy
        moved.add(id(b))
        for o in blocks:
            if id(o) not in moved and overlap(rect(o), rect(b)):
                queue.append(o)
    for s in solids:
        for o in blocks:
            if id(o) not in moved and overlap(rect(o), s):
                push_chain_more(blocks, o, moved, dx, dy)
    return [b for b in blocks if id(b) in moved]


def push_chain_more(blocks, start, moved, dx, dy):
    queue = [start]
    while queue:
        b = queue.pop()
        if id(b) in moved:
            continue
        b["x"] += dx
        b["y"] += dy
        moved.add(id(b))
        for o in blocks:
            if id(o) not in moved and overlap(rect(o), rect(b)):
                queue.append(o)


def on_beam(beam, b):
    return overlap(rect(beam), rect(b))


def hooked_block(beam, blocks):
    tip = beam["x"] + beam["w"] - 1
    for b in blocks:
        if tag(b) in ("next", "collected") and on_beam(beam, b) and b["x"] == tip - 4:
            return b
    return None


def vertical(w, dy):
    ring, beam, blocks = copy.deepcopy(w.ring), copy.deepcopy(w.beam), copy.deepcopy(w.blocks)
    carried = [b for b in blocks if on_beam(beam, b)]
    ring["y"] += dy
    beam["y"] += dy
    for b in carried:
        b["y"] += dy
    moved = {id(b) for b in carried}
    for s in [rect(ring), rect(beam)] + [rect(b) for b in carried]:
        for o in blocks:
            if id(o) not in moved and overlap(rect(o), s):
                push_chain_more(blocks, o, moved, 0, dy)
    if not w.ring_ok(ring) or not all(w.block_ok(b) for b in blocks):
        return w.ring, w.beam, w.blocks
    return ring, beam, blocks


def horizontal(w, dw):
    beam, blocks = copy.deepcopy(w.beam), copy.deepcopy(w.blocks)
    l, t, r, bt = w.field
    neww = beam["w"] + dw
    if neww < 1 or beam["x"] + neww - 1 > r:
        return w.ring, w.beam, w.blocks
    hook = hooked_block(beam, blocks)
    old_tip = beam["x"] + beam["w"] - 1
    beam["w"] = neww
    if hook is not None:
        group = [b for b in blocks if on_beam(w.beam, b)]
        push_chain(blocks, group, [], dw, 0)
    elif dw > 0:
        swept = (old_tip + 1, beam["y"], dw, beam["h"])
        targets = [b for b in blocks if tag(b) in ("next", "collected") and overlap(rect(b), swept)]
        push_chain(blocks, targets, [], dw, 0)
    if not all(w.block_ok(b) and not overlap(rect(b), rect(w.ring)) for b in blocks):
        blocks = copy.deepcopy(w.blocks)
    return w.ring, beam, blocks


def update_tags(beam, blocks, order):
    hook = hooked_block(beam, blocks)
    if hook is not None:
        others = [b for b in blocks if b is not hook and on_beam(beam, b)]
        if tag(hook) == "next" and all(tag(b) == "collected" for b in others):
            set_tag(hook, "collected")
            todo = [b for b in blocks if tag(b) == "todo"]
            todo.sort(key=lambda b: order.get(b["pixels"][0][0], 99))
            if todo:
                set_tag(todo[0], "next")
        return
    off = [b for b in blocks if tag(b) == "collected" and not on_beam(beam, b)]
    if off:
        off.sort(key=lambda b: order.get(b["pixels"][0][0], 99))
        for b in blocks:
            if tag(b) == "next":
                set_tag(b, "todo")
        set_tag(off[-1], "next")


def legend(frame, hud, colours):
    """Map block colour -> top-left of its 4x4 swatch below the HUD bar."""
    out = {}
    for c in colours:
        cells = [(x, y) for y in range(hud + 1, 64) for x in range(64) if frame[y][x] == c]
        if cells:
            out[c] = (min(x for x, _ in cells), min(y for _, y in cells))
    return out


def background(w, cover):
    l, t, r, bt = w.field
    rail_cols = set()
    for y in range(w.hud):
        for x in range(64):
            if (x, y) not in cover and frame_val(w, x, y) in (2, 3) and not (l <= x <= r and t <= y <= bt):
                rail_cols.add(x)

    def bg(x, y):
        if l <= x <= r and t <= y <= bt:
            return 4
        if x in rail_cols and t + 2 <= y <= bt - 2:
            return 2 if (y - t - 2) % 6 < 2 else 3
        return 5
    return bg


def frame_val(w, x, y):
    return w.frame[y][x]


def cells_of(o):
    return [(o["x"] + i, o["y"] + j) for j in range(o["h"]) for i in range(o["w"])]


def beam_pixel(i, j):
    return 2 if (j == 0 and i % 3 == 1) or (j == 1 and i % 3 == 0) else 1


def render(w, ring, beam, blocks, order_pos, n):
    out = [row[:] for row in w.frame]
    before = [w.ring, w.beam] + w.blocks
    cover = set()
    for o in before:
        cover.update(cells_of(o))
    bg = background(w, cover)
    for (x, y) in cover:
        if 0 <= x < 64 and 0 <= y < w.hud:
            out[y][x] = bg(x, y)
    for j in range(ring["h"]):
        for i in range(ring["w"]):
            out[ring["y"] + j][ring["x"] + i] = ring["pixels"][j][i]
    for j in range(beam["h"]):
        for i in range(beam["w"]):
            out[beam["y"] + j][beam["x"] + i] = beam_pixel(i, j)
    for b in blocks:
        c = b["pixels"][0][0]
        for (x, y) in cells_of(b):
            out[y][x] = c
    for b in blocks:
        c = b["pixels"][0][0]
        if c in order_pos:
            lx, ly = order_pos[c]
            v = 0 if tag(b) == "collected" else c
            for dy in (1, 2):
                for dx in (1, 2):
                    out[ly + dy][lx + dx] = v
    threes = max(0, (n - 1) // 3)
    row = out[w.hud]
    for k in range(64):
        x = 63 - k
        if row[x] in (2, 3):
            row[x] = 3 if k < threes else 2
    return out


def counter_from_frame(frame, hud):
    b = sum(1 for v in frame[hud] if v == 3)
    return 3 * b + 1 if b else 0


def transition_function(state, action, frame):
    if isinstance(action, dict):
        return [row[:] for row in frame]
    w = World(state, frame)
    n = _last["n"] if _last["frame"] == frame else counter_from_frame(frame, w.hud)
    n += 1
    if action == 1:
        ring, beam, blocks = vertical(w, -STEP)
    elif action == 2:
        ring, beam, blocks = vertical(w, STEP)
    elif action == 4:
        ring, beam, blocks = horizontal(w, STEP)
    elif action == 3:
        ring, beam, blocks = horizontal(w, -STEP)
    else:
        ring, beam, blocks = w.ring, w.beam, w.blocks
    blocks = copy.deepcopy(blocks)
    colours = [b["pixels"][0][0] for b in w.blocks]
    pos = legend(frame, w.hud, colours)
    order = {c: p[0] for c, p in pos.items()}
    update_tags(beam, blocks, order)
    out = render(w, ring, beam, blocks, pos, n)
    _last["frame"] = out
    _last["n"] = n
    return out
