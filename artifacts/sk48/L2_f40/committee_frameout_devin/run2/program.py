# Mechanics: ring (player) on a left rail; beam (arm) from ring.x+5. A1/A2 move ring+beam y-/+6 carrying on-beam blocks,
#  chain-pushing hit blocks; a block leaving the playfield (largest colour-4 rect) cancels. A4/A3 beam w +/-6 inside field;
#  hooked next/collected block (right edge = tip-1) drags on-beam blocks, else extension pushes swept blocks; stuck group ->
#  beam moves alone. Clicks no-op. HUD bar: rightmost 2->3 every 3rd non-click action (hidden counter, continuity-gated).
#  Legend centre 0 while collected. Unconfirmed: A5/A7 (assumed counted no-ops), empty bar, rail extent outside data.

STEP = 6
_mem = {"frame": None, "n": None}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def inside(r, pf):
    return r[0] >= pf[0] and r[1] >= pf[1] and r[0] + r[2] <= pf[0] + pf[2] and r[1] + r[3] <= pf[1] + pf[3]


def colour_of(o):
    cnt = {}
    for row in o["pixels"]:
        for v in row:
            cnt[v] = cnt.get(v, 0) + 1
    return max(cnt, key=cnt.get)


def playfield(frame, objs):
    H, W = len(frame), len(frame[0])
    cov = set((o["x"] + i, o["y"] + j) for o in objs for j in range(o["h"]) for i in range(o["w"]))
    seen, best = set(), None
    for y in range(H):
        for x in range(W):
            if frame[y][x] != 4 or (x, y) in seen:
                continue
            stack, comp = [(x, y)], []
            seen.add((x, y))
            while stack:
                cx, cy = stack.pop()
                if frame[cy][cx] == 4:
                    comp.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen and (
                            frame[ny][nx] == 4 or (nx, ny) in cov):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            if best is None or len(comp) > len(best):
                best = comp
    xs = [c[0] for c in best]
    ys = [c[1] for c in best]
    return (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)


def push_chain(moving, others, dx, dy, solids):
    """Move `moving` blocks by (dx,dy); blocks hit by moved blocks or solids get pushed too."""
    moved = set(id(b) for b in moving)
    group = list(moving)
    changed = True
    while changed:
        changed = False
        rects = [(b["x"] + dx, b["y"] + dy, b["w"], b["h"]) for b in group] + solids
        for o in others:
            if id(o) in moved:
                continue
            if any(overlap(rect(o), r) for r in rects):
                moved.add(id(o))
                group.append(o)
                changed = True
                break
    return group


def beam_rect(beam, w=None):
    return (beam["x"], beam["y"], beam["w"] if w is None else w, beam["h"])


def on_beam(b, beam):
    return overlap(rect(b), beam_rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    return [b for b in blocks if on_beam(b, beam) and b["x"] + b["w"] == tip
            and ("next" in b["tags"] or "collected" in b["tags"])]


def step_objects(ring, beam, blocks, act, pf):
    """Return new (ring y, beam w, {id(block): (x, y)}) ."""
    pos = {id(b): (b["x"], b["y"]) for b in blocks}
    if act in (1, 2):
        dy = -STEP if act == 1 else STEP
        ny = ring["y"] + dy
        if ny < pf[1] or ny + ring["h"] > pf[1] + pf[3]:
            return ring["y"], beam["w"], pos
        carried = [b for b in blocks if on_beam(b, beam)]
        solids = [(ring["x"], ny, ring["w"], ring["h"]), (beam["x"], beam["y"] + dy, beam["w"], beam["h"])]
        group = push_chain(carried, blocks, 0, dy, solids)
        if not all(inside((b["x"], b["y"] + dy, b["w"], b["h"]), pf) for b in group):
            return ring["y"], beam["w"], pos
        for b in group:
            pos[id(b)] = (b["x"], b["y"] + dy)
        return ny, beam["w"], pos
    if act in (3, 4):
        dw = STEP if act == 4 else -STEP
        nw = beam["w"] + dw
        if nw < 1 or beam["x"] + nw > pf[0] + pf[2]:
            return ring["y"], beam["w"], pos
        if hooked(blocks, beam):
            group = push_chain([b for b in blocks if on_beam(b, beam)], blocks, dw, 0, [])
        elif dw > 0:
            swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
            group = push_chain([], blocks, dw, 0, [swept])
        else:
            group = []
        if group and all(inside((b["x"] + dw, b["y"], b["w"], b["h"]), pf) for b in group):
            for b in group:
                pos[id(b)] = (b["x"] + dw, b["y"])
        return ring["y"], nw, pos
    return ring["y"], beam["w"], pos


def update_tags(blocks, beam):
    """Return set of ids of collected blocks after the move (blocks already at new positions)."""
    state = {id(b): ("collected" if "collected" in b["tags"] else "next" if "next" in b["tags"] else "todo")
             for b in blocks}
    hk = hooked(blocks, beam)
    onb = [b for b in blocks if on_beam(b, beam)]
    for b in hk:
        if state[id(b)] == "next" and all(state[id(o)] == "collected" for o in onb if o is not b):
            state[id(b)] = "collected"
            todo = [o for o in blocks if state[id(o)] == "todo"]
            if todo:
                state[id(max(todo, key=lambda o: o["x"]))] = "next"
    if not hk:
        for b in blocks:
            if state[id(b)] == "collected" and not on_beam(b, beam):
                for o in blocks:
                    if state[id(o)] == "next":
                        state[id(o)] = "todo"
                state[id(b)] = "next"
    return set(k for k, v in state.items() if v == "collected")


def background(frame, pf, objs):
    covered = set()
    for o in objs:
        for j in range(o["h"]):
            for i in range(o["w"]):
                covered.add((o["x"] + i, o["y"] + j))
    top, bot = pf[1] + 2, pf[1] + pf[3] - 3
    rail = {}
    for (x, y) in ((x, y) for y in range(top, bot + 1) for x in range(len(frame[0]))):
        if (x < pf[0] or x >= pf[0] + pf[2]) and (x, y) not in covered and frame[y][x] in (2, 3):
            rail.setdefault(x, {})[(y - top) % 6] = frame[y][x]

    def bg(x, y):
        if pf[0] <= x < pf[0] + pf[2] and pf[1] <= y < pf[1] + pf[3]:
            return 4
        if x in rail and top <= y <= bot and (y - top) % 6 in rail[x]:
            return rail[x][(y - top) % 6]
        return 5
    return bg


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def paint(frame, x, y, pix):
    for j, row in enumerate(pix):
        for i, v in enumerate(row):
            if 0 <= y + j < len(frame) and 0 <= x + i < len(frame[0]) and v >= 0:
                frame[y + j][x + i] = v


def bar_row(frame, pf):
    for y in range(pf[1] + pf[3], len(frame)):
        if all(v in (2, 3) for v in frame[y]):
            return y
    return None


def transition_function(state, action, frame):
    out = [row[:] for row in frame]
    if isinstance(action, dict):
        _mem["frame"] = out
        return out
    act = action
    pf = playfield(frame, state)
    ring = next(o for o in state if o["type"] == "player")
    beam = next(o for o in state if o["type"] == "arm")
    blocks = [o for o in state if o["type"] == "block"]
    bg = background(frame, pf, state)

    ny, nw, pos = step_objects(ring, beam, blocks, act, pf)
    dy = ny - ring["y"]
    nbeam = dict(beam, y=beam["y"] + dy, w=nw)
    nblocks = [dict(b, x=pos[id(b)][0], y=pos[id(b)][1]) for b in blocks]
    collected = update_tags(nblocks, nbeam)
    colours = {colour_of(b): (id(nb) in collected) for b, nb in zip(blocks, nblocks)}

    for o in state:
        for j in range(o["h"]):
            for i in range(o["w"]):
                x, y = o["x"] + i, o["y"] + j
                if 0 <= y < len(out) and 0 <= x < len(out[0]):
                    out[y][x] = bg(x, y)
    paint(out, ring["x"], ny, ring["pixels"])
    paint(out, nbeam["x"], nbeam["y"], beam_pixels(nw))
    for b, nb in zip(blocks, nblocks):
        c = colour_of(b)
        paint(out, nb["x"], nb["y"], [[c] * b["w"] for _ in range(b["h"])])

    by = bar_row(frame, pf)
    if by is not None:
        d = sum(1 for v in frame[by] if v == 3)
        if _mem["frame"] is not None and _mem["frame"] == frame and _mem["n"] is not None:
            n = _mem["n"]
        else:
            n = 3 * d + 1 if d else 0
        n += 1
        if n >= 4 and (n - 1) % 3 == 0:
            twos = [x for x, v in enumerate(frame[by]) if v == 2]
            if twos:
                out[by][max(twos)] = 3
        _mem["n"] = n
        for c, col in colours.items():
            cells = [(x, y) for y in range(by + 1, len(frame)) for x in range(len(frame[0]))
                     if frame[y][x] == c]
            if not cells:
                continue
            x0, y0 = min(p[0] for p in cells), min(p[1] for p in cells)
            for j in (1, 2):
                for i in (1, 2):
                    out[y0 + j][x0 + i] = 0 if col else c
    _mem["frame"] = out
    return out
