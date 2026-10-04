# Mechanics: crane ring (player) moves y-+6 (A1/A2) carrying blocks on the beam and pushing blocks hit
# (push chain; the whole move cancels if a block leaves the colour-4 field). A4/A3 beam w+-6: hooked
# next/collected block at tip-4 drags on-beam blocks, else extension pushes swept blocks; a blocked group
# leaves the beam moving alone. Tags collect/revert drive legend centres; clicks are no-ops.
# Hidden: HUD bar 3s=(n-1)//3, n = non-click actions, carried on frame continuity (fallback 3*bars+1 or 0).
import copy

_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


class World:
    def __init__(self, frame):
        self.hud = next(y for y in range(64) if all(v not in (4, 5) for v in frame[y]))
        cells = [(x, y) for y in range(self.hud) for x in range(64) if frame[y][x] == 4]
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        self.l, self.r, self.t, self.b = min(xs), max(xs), min(ys), max(ys)

    def in_field(self, r):
        return r[0] >= self.l and r[0] + r[2] - 1 <= self.r and r[1] >= self.t and r[1] + r[3] - 1 <= self.b

    def bg(self, x, y, ring):
        if self.l <= x <= self.r and self.t <= y <= self.b:
            return 4
        if ring["x"] + 2 <= x <= ring["x"] + 3 and self.t + 2 <= y <= self.b - 2:
            return 2 if (y - self.t - 2) % 6 < 2 else 3
        return 5


def block_ok(w, b, ring):
    return w.in_field(rect(b)) and not overlap(rect(b), rect(ring))


def push_chain(blocks, movers, solids, dx, dy):
    moved = set(movers)
    while True:
        rs = [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, 4, 4) for i in moved] + solids
        new = {i for i, b in enumerate(blocks) if i not in moved and any(overlap(rect(b), r) for r in rs)}
        if not new:
            return moved
        moved |= new


def step(state, action, world, names):
    st = copy.deepcopy(state)
    ring = next(o for o in st if o["type"] == "player")
    beam = next(o for o in st if o["type"] == "arm")
    blocks = [o for o in st if o["type"] == "block"]
    on_beam = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
    if action in (1, 2):
        d = -6 if action == 1 else 6
        nring = (ring["x"], ring["y"] + d, ring["w"], ring["h"])
        nbeam = (beam["x"], beam["y"] + d, beam["w"], beam["h"])
        if nring[1] < world.t or nring[1] + nring[3] - 1 > world.b:
            return st
        moved = push_chain(blocks, on_beam, [nring, nbeam], 0, d)
        trial = [dict(b, y=b["y"] + d) if i in moved else b for i, b in enumerate(blocks)]
        nr = dict(ring, y=ring["y"] + d)
        if all(block_ok(world, trial[i], nr) for i in moved):
            ring["y"] += d
            beam["y"] += d
            for i in moved:
                blocks[i]["y"] += d
    elif action in (3, 4):
        d = 6 if action == 4 else -6
        nw = beam["w"] + d
        if nw < 1 or beam["x"] + nw - 1 > world.r:
            return st
        tip = beam["x"] + beam["w"] - 1
        hooked = any(i in on_beam and blocks[i]["x"] == tip - 4 and
                     ("next" in blocks[i]["tags"] or "collected" in blocks[i]["tags"])
                     for i in range(len(blocks)))
        moved = set()
        if hooked:
            moved = push_chain(blocks, on_beam, [], d, 0)
        elif d > 0:
            strip = (beam["x"] + beam["w"], beam["y"], d, beam["h"])
            moved = push_chain(blocks, [], [strip], d, 0)
        trial = [dict(b, x=b["x"] + d) if i in moved else b for i, b in enumerate(blocks)]
        if all(block_ok(world, trial[i], ring) for i in moved):
            for i in moved:
                blocks[i]["x"] += d
        beam["w"] = nw
    update_tags(st, names)
    return st


def legend_order(frame, world, blocks):
    pos = {}
    for b in blocks:
        c = b["pixels"][0][0]
        xs = [x for y in range(world.hud + 1, 64) for x in range(64) if frame[y][x] == c]
        pos[b["name"]] = min(xs) if xs else 99
    return sorted(blocks, key=lambda b: pos[b["name"]])


def update_tags(st, names):
    beam = next(o for o in st if o["type"] == "arm")
    blocks = [o for o in st if o["type"] == "block"]
    tip = beam["x"] + beam["w"] - 1
    on = [b for b in blocks if overlap(rect(b), rect(beam))]
    hook = [b for b in on if b["x"] == tip - 4 and ("next" in b["tags"] or "collected" in b["tags"])]
    byname = {b["name"]: b for b in blocks}
    order = [byname[k] for k in names]

    def settag(b, t):
        b["tags"] = [x for x in b["tags"] if x not in ("todo", "next", "collected")] + [t]

    for b in blocks:
        if "collected" in b["tags"] and not hook and b not in on:
            for o in blocks:
                if "next" in o["tags"]:
                    settag(o, "todo")
            settag(b, "next")
    for b in hook:
        if "next" in b["tags"] and all("collected" in o["tags"] for o in on if o is not b):
            settag(b, "collected")
            todo = [o for o in order if "todo" in o["tags"]]
            if todo:
                settag(todo[0], "next")


def draw(frame, o):
    if o["type"] == "arm":
        for j in range(o["h"]):
            for i in range(o["w"]):
                frame[o["y"] + j][o["x"] + i] = 2 if (i + j) % 3 == 1 else 1
        return
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            if 0 <= o["y"] + j < 64 and 0 <= o["x"] + i < 64:
                frame[o["y"] + j][o["x"] + i] = v


def transition_function(state, action, frame):
    world = World(frame)
    is_click = isinstance(action, dict)
    if _mem["frame"] is not None and frame == _mem["frame"]:
        n = _mem["n"]
    else:
        threes = sum(1 for v in frame[world.hud] if v == 3)
        n = 3 * threes + 1 if threes else 0
    blocks = [o for o in state if o["type"] == "block"]
    order = legend_order(frame, world, blocks)
    state = copy.deepcopy(state)
    ordered_names = [b["name"] for b in order]
    new = state if is_click else step(state, action, world, ordered_names)
    if not is_click:
        n += 1
    out = [row[:] for row in frame]
    ring0 = next(o for o in state if o["type"] == "player")
    for o in state:
        for j in range(o["h"]):
            for i in range(o["w"]):
                x, y = o["x"] + i, o["y"] + j
                if 0 <= x < 64 and 0 <= y < 64:
                    out[y][x] = world.bg(x, y, ring0)
    for o in sorted(new, key=lambda o: o["layer"]):
        draw(out, o)
    for b in order:
        c = b["pixels"][0][0]
        cells = [(x, y) for y in range(world.hud + 1, 64) for x in range(64) if frame[y][x] == c]
        if not cells:
            continue
        x0, y0 = min(p[0] for p in cells), min(p[1] for p in cells)
        col = 0 if "collected" in next(o for o in new if o["name"] == b["name"])["tags"] else c
        for j in (1, 2):
            for i in (1, 2):
                out[y0 + j][x0 + i] = col
    k = max(0, (n - 1) // 3)
    for x in range(64):
        if frame[world.hud][x] in (2, 3):
            out[world.hud][x] = 3 if x >= 64 - k else 2
    _mem["frame"] = out
    _mem["n"] = n
    return out
