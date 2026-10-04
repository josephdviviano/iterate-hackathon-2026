# Mechanics: ring (player) + beam (arm) move vertically by 6 carrying blocks on the beam and pushing
# blocks hit by the moved beam (push chain); cancelled if any block would leave the colour-4 field.
# A4/A3 grow/shrink the beam by 6; a block at the tip (x == end-5, any tag) hooks every fully-on-beam
# block to follow the tip; an unhooked extend pushes blocks in the swept strip; blocked -> beam alone.
# Tags: hooked next with only collected on-beam -> collected; off-beam collected reverts. HUD bar = (n-1)//3.
_MEMO = {}

def _key(frame):
    return tuple(tuple(r) for r in frame)

def _overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]

class World:
    def __init__(self, frame):
        self.hud = next(y for y in range(64) if all(v not in (4, 5) for v in frame[y]))
        cells = [(x, y) for y in range(self.hud) for x in range(64) if frame[y][x] == 4]
        self.left = min(c[0] for c in cells)
        self.right = max(c[0] for c in cells)
        self.top = min(c[1] for c in cells)
        self.bot = max(c[1] for c in cells)

    def inside(self, r):
        return (r[0] >= self.left and r[0] + r[2] - 1 <= self.right
                and r[1] >= self.top and r[1] + r[3] - 1 <= self.bot)

def block_rect(b):
    return [b["x"], b["y"], b["w"], b["h"]]

def push_chain(blocks, movers, pushers, dx, dy):
    """Move `movers` by (dx,dy); any other block overlapped by a moved rect or a pusher joins."""
    moved = set(movers)
    frontier = [block_rect(blocks[i]) for i in movers]
    frontier = [[r[0] + dx, r[1] + dy, r[2], r[3]] for r in frontier] + list(pushers)
    while frontier:
        r = frontier.pop()
        for j, b in enumerate(blocks):
            if j not in moved and _overlap(r, block_rect(b)):
                moved.add(j)
                br = block_rect(b)
                frontier.append([br[0] + dx, br[1] + dy, br[2], br[3]])
    return moved

def apply_move(world, ring, blocks, moved, dx, dy):
    new = []
    for j, b in enumerate(blocks):
        nb = dict(b)
        if j in moved:
            nb["x"] += dx
            nb["y"] += dy
        new.append(nb)
    ring_r = [ring["x"], ring["y"], ring["w"], ring["h"]]
    for j in moved:
        r = block_rect(new[j])
        if not world.inside(r) or _overlap(r, ring_r):
            return None
    return new

def beam_rect(beam):
    return [beam["x"], beam["y"], beam["w"], beam["h"]]

def on_beam_rows(beam, b):
    return b["y"] < beam["y"] + beam["h"] and beam["y"] < b["y"] + b["h"]

def fully_on_beam(beam, b):
    return on_beam_rows(beam, b) and b["x"] >= beam["x"] and b["x"] + b["w"] <= beam["x"] + beam["w"]

def hooked(beam, b):
    return on_beam_rows(beam, b) and b["x"] == beam["x"] + beam["w"] - 5

def status(b):
    return b["tags"][-1]

def set_status(b, s):
    b["tags"] = list(b["tags"][:-1]) + [s]

def vertical(world, ring, beam, blocks, dy):
    nring = dict(ring, y=ring["y"] + dy)
    if nring["y"] < world.top or nring["y"] + nring["h"] - 1 > world.bot:
        return ring, beam, blocks
    nbeam = dict(beam, y=beam["y"] + dy)
    carried = [j for j, b in enumerate(blocks) if _overlap(beam_rect(beam), block_rect(b))]
    pushers = [beam_rect(nbeam), [nring["x"], nring["y"], nring["w"], nring["h"]]]
    moved = push_chain(blocks, carried, pushers, 0, dy)
    new = apply_move(world, nring, blocks, moved, 0, dy)
    if new is None:
        return ring, beam, blocks
    return nring, nbeam, new

def horizontal(world, ring, beam, blocks, dw):
    nw = beam["w"] + dw
    if nw < 1 or beam["x"] + nw - 1 > world.right:
        return beam, blocks
    nbeam = dict(beam, w=nw)
    if any(hooked(beam, b) for b in blocks):
        movers = [j for j, b in enumerate(blocks) if fully_on_beam(beam, b)]
        pushers = []
        if dw > 0:
            pushers = [[beam["x"] + beam["w"], beam["y"], dw, beam["h"]]]
        moved = push_chain(blocks, movers, pushers, dw, 0)
    elif dw > 0:
        moved = push_chain(blocks, [], [[beam["x"] + beam["w"], beam["y"], dw, beam["h"]]], dw, 0)
    else:
        moved = set()
    new = apply_move(world, ring, blocks, moved, dw, 0) if moved else blocks
    if new is None:
        return nbeam, blocks
    return nbeam, new

def update_tags(beam, blocks, legend):
    blocks = [dict(b, tags=list(b["tags"])) for b in blocks]
    order = sorted(range(len(blocks)), key=lambda j: legend.get(blocks[j]["tags"][1], (99, 99)))
    nxt = [j for j in order if status(blocks[j]) == "next"]
    if nxt:
        j = nxt[0]
        others = [b for k, b in enumerate(blocks) if k != j and fully_on_beam(beam, b)]
        if hooked(beam, blocks[j]) and all(status(b) == "collected" for b in others):
            set_status(blocks[j], "collected")
            for k in order:
                if status(blocks[k]) == "todo":
                    set_status(blocks[k], "next")
                    break
            return blocks
    off = [j for j in order if status(blocks[j]) == "collected" and not fully_on_beam(beam, blocks[j])]
    if off:
        for j in nxt:
            set_status(blocks[j], "todo")
        set_status(blocks[off[-1]], "next")
    return blocks

def legend_squares(frame, world, colours):
    sq = {}
    for c in colours:
        pts = [(x, y) for y in range(world.hud + 1, 64) for x in range(64) if frame[y][x] == int(c)]
        if pts:
            sq[c] = (min(p[0] for p in pts), min(p[1] for p in pts))
    return sq

def background(world, ring, x, y):
    if world.left <= x <= world.right and world.top <= y <= world.bot:
        return 4
    if x in (ring["x"] + 2, ring["x"] + 3) and world.top + 2 <= y <= world.bot - 2:
        return 2 if (y - world.top - 2) % 6 < 2 else 3
    return 5

def cells_of(o):
    if o["type"] == "block":
        c = int(o["tags"][1])
        return [(o["x"] + i, o["y"] + j, c) for j in range(o["h"]) for i in range(o["w"])]
    if o["type"] == "arm":
        return [(o["x"] + i, o["y"] + j, 2 if (i + j) % 3 == 1 else 1)
                for j in range(o["h"]) for i in range(o["w"])]
    out = []
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            if v is not None and v >= 0:
                out.append((o["x"] + i, o["y"] + j, v))
    return out

def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    key = _key(frame)
    world = World(frame)
    hud = frame[world.hud]
    threes = 0
    while threes < 64 and hud[63 - threes] == 3:
        threes += 1
    n = _MEMO.get(key, 3 * threes + 1 if threes else 0)
    if isinstance(action, dict) or action not in (1, 2, 3, 4):
        _MEMO[key] = n
        return out
    ring = next(o for o in state if o["type"] == "player")
    beam = next(o for o in state if o["type"] == "arm")
    blocks = [o for o in state if o["type"] == "block"]
    legend = legend_squares(frame, world, [b["tags"][1] for b in blocks])
    nring, nbeam, nblocks = ring, beam, blocks
    if action in (1, 2):
        nring, nbeam, nblocks = vertical(world, ring, beam, blocks, -6 if action == 1 else 6)
    else:
        nbeam, nblocks = horizontal(world, ring, beam, blocks, 6 if action == 4 else -6)
    nblocks = update_tags(nbeam, nblocks, legend)
    for o in [ring, beam] + blocks:
        for x, y, _ in cells_of(o):
            if 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = background(world, ring, x, y)
    for o in sorted([nring, nbeam] + nblocks, key=lambda o: o["layer"]):
        for x, y, v in cells_of(o):
            if 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = v
    for b in nblocks:
        c = b["tags"][1]
        if c in legend:
            lx, ly = legend[c]
            fill = 0 if status(b) == "collected" else int(c)
            for yy in (ly + 1, ly + 2):
                for xx in (lx + 1, lx + 2):
                    out[yy][xx] = fill
    n += 1
    bar = max(0, (n - 1) // 3)
    for k in range(min(bar, 64)):
        if out[world.hud][63 - k] == 2:
            out[world.hud][63 - k] = 3
    _MEMO[_key(out)] = n
    return out
