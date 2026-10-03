# Mechanics: ring (player) A1/A2 moves y-/+6 with the beam, carrying blocks fully on the beam and pushing blocks hit
#   (push chain); any block leaving the colour-4 field or ring leaving field rows cancels the move. Beam (arm) A4/A3 w+-6
#   (1..field right edge): blocks fully on the beam move with the tip, extension also pushes blocks in the swept strip;
#   if that group would leave the field or enter the ring, the beam changes alone. Tags: hooked next with all other
#   on-beam blocks collected -> collected; collected off beam -> next. HUD bar 3s=(n-1)//3, n=non-click actions (hidden,
#   gated on returned frames; fallback 3*bar+1). Hypothesis: todo/next tags never change physics (t69 = fully-on-beam).
import copy

STEP = 6


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def tagset(o):
    return set(o.get("tags", []))


class World:
    def __init__(self, frame):
        self.frame = frame
        H, W = len(frame), len(frame[0])
        self.hud = next(y for y in range(H) if all(v not in (4, 5) for v in frame[y]))
        cells = [(x, y) for y in range(self.hud) for x in range(W) if frame[y][x] == 4]
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        self.left, self.right = min(xs), max(xs)
        self.top, self.bot = min(ys), max(ys)

    def in_field(self, r):
        return (r[0] >= self.left and r[0] + r[2] - 1 <= self.right
                and r[1] >= self.top and r[1] + r[3] - 1 <= self.bot)


def find(state, typ):
    return [o for o in state if o.get("type") == typ]


def on_beam(b, beam):
    return (b["y"] <= beam["y"] and b["y"] + b["h"] >= beam["y"] + beam["h"]
            and b["x"] >= beam["x"] and b["x"] + b["w"] <= beam["x"] + beam["w"])


def block_ok(world, ring, b):
    return world.in_field(rect(b)) and not overlap(rect(b), rect(ring))


def push_chain(blocks, movers, solids, dx, dy):
    """Move movers by (dx,dy); any block hit by a moved block or a solid rect is pushed too."""
    moved = set()
    queue = list(movers)
    for b in blocks:
        if b["name"] not in moved and any(overlap(rect(b), s) for s in solids) and b not in queue:
            queue.append(b)
    while queue:
        b = queue.pop(0)
        if b["name"] in moved:
            continue
        moved.add(b["name"])
        b["x"] += dx
        b["y"] += dy
        for o in blocks:
            if o["name"] not in moved and o not in queue and overlap(rect(o), rect(b)):
                queue.append(o)
    return [b for b in blocks if b["name"] in moved]


def beam_pixels(w, h):
    return [[2 if (i + j) % 3 == 1 else 1 for i in range(w)] for j in range(h)]


def step_vertical(world, state, dy):
    ring, beam = find(state, "player")[0], find(state, "arm")[0]
    blocks = find(state, "block")
    carried = [b for b in blocks if on_beam(b, beam)]
    ny = ring["y"] + dy
    if ny < world.top or ny + ring["h"] - 1 > world.bot:
        return False
    ring["y"] = ny
    beam["y"] += dy
    moved = push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(world, ring, b) for b in moved)


def step_horizontal(world, state, dw):
    ring, beam = find(state, "player")[0], find(state, "arm")[0]
    blocks = find(state, "block")
    nw = beam["w"] + dw
    if nw < 1 or beam["x"] + nw - 1 > world.right:
        return False
    old_blocks = copy.deepcopy(blocks)
    tip = beam["x"] + beam["w"]
    movers = [b for b in blocks if on_beam(b, beam)]
    solids = []
    if dw > 0:
        solids = [(tip, beam["y"], dw, beam["h"])]
    moved = push_chain(blocks, movers, solids, dw, 0)
    if not all(block_ok(world, ring, b) for b in moved):
        for b, ob in zip(blocks, old_blocks):
            b["x"], b["y"] = ob["x"], ob["y"]
    beam["w"] = nw
    beam["pixels"] = beam_pixels(nw, beam["h"])
    return True


def set_status(b, status):
    b["tags"] = [t for t in b["tags"] if t not in ("todo", "next", "collected")] + [status]


def update_tags(state, legend_order):
    beam = find(state, "arm")[0]
    blocks = find(state, "block")
    tip = beam["x"] + beam["w"] - 1
    onb = [b for b in blocks if on_beam(b, beam)]
    nxt = [b for b in blocks if "next" in tagset(b)]
    if nxt:
        n = nxt[0]
        if n in onb and n["x"] == tip - 4 and all("collected" in tagset(o) for o in onb if o is not n):
            set_status(n, "collected")
            rest = [b for b in sorted(blocks, key=legend_order) if "todo" in tagset(b)]
            if rest:
                set_status(rest[0], "next")
            return
    for c in blocks:
        if "collected" in tagset(c) and c not in onb:
            for n in blocks:
                if "next" in tagset(n):
                    set_status(n, "todo")
            set_status(c, "next")
            return


def block_colour(b):
    for row in b["pixels"]:
        for v in row:
            return v
    return None


def legend_boxes(world, frame, state):
    boxes = {}
    for b in find(state, "block"):
        c = block_colour(b)
        cells = [(x, y) for y in range(world.hud + 1, len(frame)) for x in range(len(frame[0])) if frame[y][x] == c]
        if cells:
            boxes[b["name"]] = (min(p[0] for p in cells), min(p[1] for p in cells),
                                max(p[0] for p in cells), max(p[1] for p in cells))
    return boxes


def rail_cols(world, frame, state):
    covered = set()
    for o in state:
        for y in range(o["y"], o["y"] + o["h"]):
            for x in range(o["x"], o["x"] + o["w"]):
                covered.add((x, y))
    cols = set()
    for y in range(world.top, world.bot + 1):
        for x in range(len(frame[0])):
            if (world.left <= x <= world.right) or (x, y) in covered:
                continue
            if frame[y][x] in (2, 3):
                cols.add(x)
    return cols


def background(world, rails, x, y):
    if world.left <= x <= world.right and world.top <= y <= world.bot:
        return 4
    if x in rails and world.top + 2 <= y <= world.bot - 2:
        return 2 if (y - world.top - 2) % 6 < 2 else 3
    return 5


def draw_obj(out, o):
    if not o.get("visible", True):
        return
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            y, x = o["y"] + j, o["x"] + i
            if 0 <= y < len(out) and 0 <= x < len(out[0]) and v is not None and v >= 0:
                out[y][x] = v


def render(world, frame, before, after, rails):
    out = [list(r) for r in frame]
    for o in before:
        for y in range(o["y"], o["y"] + o["h"]):
            for x in range(o["x"], o["x"] + o["w"]):
                if y < world.hud:
                    out[y][x] = background(world, rails, x, y)
    for o in sorted(after, key=lambda o: o.get("layer", 0)):
        draw_obj(out, o)
    return out


def render_legend(out, boxes, after):
    for b in find(after, "block"):
        if b["name"] not in boxes:
            continue
        x0, y0, x1, y1 = boxes[b["name"]]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        v = 0 if "collected" in tagset(b) else block_colour(b)
        for y in (cy, cy + 1):
            for x in (cx, cx + 1):
                out[y][x] = v


def render_bar(out, world, n):
    row = out[world.hud]
    filled = max(0, (n - 1) // 3)
    W = len(row)
    for x in range(W):
        row[x] = 3 if x >= W - filled else 2


def bar_count(frame, world):
    return sum(1 for v in frame[world.hud] if v == 3)


_seen = {}


def frame_key(f):
    return tuple(tuple(r) for r in f)


def transition_function(state, action, frame):
    world = World(frame)
    key = frame_key(frame)
    if key in _seen:
        n = _seen[key]
    else:
        b = bar_count(frame, world)
        n = 3 * b + 1 if b else 0
    after = copy.deepcopy(state)
    boxes = legend_boxes(world, frame, state)
    order = lambda o: boxes.get(o["name"], (99,))[0]
    aid = action if isinstance(action, int) else action.get("action_id")
    if aid in (1, 2, 3, 4):
        trial = copy.deepcopy(state)
        if aid in (1, 2):
            ok = step_vertical(world, trial, -STEP if aid == 1 else STEP)
        else:
            ok = step_horizontal(world, trial, STEP if aid == 4 else -STEP)
        if ok:
            after = trial
            update_tags(after, order)
        n += 1
    rails = rail_cols(world, frame, state)
    out = render(world, frame, state, after, rails)
    render_legend(out, boxes, after)
    render_bar(out, world, n)
    out = [[int(v) for v in r] for r in out]
    _seen[frame_key(out)] = n
    return out
