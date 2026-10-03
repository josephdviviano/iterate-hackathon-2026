# Mechanics: crane ring+beam (A1/A2 move y-/+6 carrying on-beam blocks, push chains; cancel if any block leaves y[2,58]);
# A4/A3 beam w +-6 in [1,43]; hooked next/collected block (x == tip-4) drags on-beam blocks, unhooked extend pushes swept
# next/collected blocks; blocked group (x>48 or into ring) -> beam alone. Tags follow HUD legend order (collect/revert).
# Frame: background under vacated cells = nearest uncovered cell in column; rail = period-6 2/3 pattern ending in 2-pairs.
# HUD: row-53 bar gains one 3-cell (from the right) every 3 non-click actions (hidden counter); clicks are no-ops.
import copy

BW, BMAX, YMIN, YMAX, XMAX = 6, 43, 2, 58, 48
_mem = {"frame": None, "k": None}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def kind(o):
    return o["tags"][-1]


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def push_chain(blocks, movers, solids, dx, dy):
    moved = set(movers)
    while True:
        rs = [(b["x"] + dx, b["y"] + dy, b["w"], b["h"]) for i, b in enumerate(blocks) if i in moved] + solids
        add = {i for i, b in enumerate(blocks) if i not in moved and any(overlap(rect(b), r) for r in rs)}
        if not add:
            return moved
        moved |= add


def block_ok(b, ring):
    return YMIN <= b["y"] <= YMAX and ring["x"] + ring["w"] <= b["x"] <= XMAX


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    return [i for i, b in enumerate(blocks) if kind(b) in ("next", "collected")
            and b["x"] == tip - 4 and on_beam(b, beam)]


def try_move(blocks, idx, dx, dy, ring):
    trial = copy.deepcopy(blocks)
    for i in idx:
        trial[i]["x"] += dx
        trial[i]["y"] += dy
    if all(block_ok(trial[i], ring) for i in idx):
        return trial
    return None


def step(ring, beam, blocks, action):
    if action in (1, 2):
        dy = -BW if action == 1 else BW
        nr, nb = dict(ring), dict(beam)
        nr["y"] += dy
        nb["y"] += dy
        if not (0 <= nr["y"] <= YMAX):
            return ring, beam, blocks
        carried = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
        moved = push_chain(blocks, carried, [rect(nr), rect(nb)], 0, dy)
        trial = try_move(blocks, moved, 0, dy, nr)
        if trial is None:
            return ring, beam, blocks
        return nr, nb, trial
    if action in (3, 4):
        dw = BW if action == 4 else -BW
        w = beam["w"] + dw
        if not (1 <= w <= BMAX):
            return ring, beam, blocks
        nb = dict(beam, w=w, pixels=beam_pixels(w))
        hk = hooked(blocks, beam)
        if hk:
            movers = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
        elif dw > 0:
            swept = (beam["x"] + beam["w"], beam["y"], dw, beam["h"])
            movers = {i for i, b in enumerate(blocks)
                      if kind(b) in ("next", "collected") and overlap(rect(b), swept)}
        else:
            movers = set()
        if movers:
            moved = push_chain(blocks, movers, [], dw, 0)
            trial = try_move(blocks, moved, dw, 0, ring)
            if trial is not None:
                blocks = trial
        return ring, nb, blocks
    return ring, beam, blocks


def update_tags(blocks, beam, order):
    def tag(b, t):
        b["tags"] = b["tags"][:-1] + [t]
    seq = [b for o in order for b in blocks if b["name"] == o] + [b for b in blocks if b["name"] not in order]
    hk = hooked(blocks, beam)
    nxt = [b for b in seq if kind(b) == "next"]
    onb = [b for b in blocks if on_beam(b, beam)]
    if nxt and blocks.index(nxt[0]) in hk and all(kind(b) == "collected" for b in onb if b is not nxt[0]):
        tag(nxt[0], "collected")
        todo = [b for b in seq if kind(b) == "todo"]
        if todo:
            tag(todo[0], "next")
    elif not hk:
        col = [b for b in seq if kind(b) == "collected"]
        if col and not on_beam(col[-1], beam):
            for b in nxt:
                tag(b, "todo")
            tag(col[-1], "next")


def cells(o):
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            if v >= 0:
                yield o["x"] + i, o["y"] + j, v


def is_rail_seed(frame, cov, x, y):
    pat = [2, 2, 3, 3, 3, 3, 2]
    return y + 6 < len(frame) and all(not cov[y + k][x] and frame[y + k][x] == pat[k] for k in range(7))


def rail_runs(frame, cov):
    runs = {}
    H, W = len(frame), len(frame[0])
    for x in range(W):
        for y in range(H):
            if is_rail_seed(frame, cov, x, y):
                ok = lambda yy: 0 <= yy < H and (cov[yy][x] or frame[yy][x] in (2, 3))
                a, b = y, y
                while ok(a - 1):
                    a -= 1
                while ok(b + 1):
                    b += 1
                t = a + (y - a) % 6
                e = b - (b - (y + 1)) % 6
                runs[x] = (t, e, y)
                break
    return runs


def background(frame, cov, x, y, runs):
    if x in runs:
        t, e, ph = runs[x]
        if t <= y <= e:
            return 2 if (y - ph) % 6 in (0, 1) else 3
    H = len(frame)
    for d in range(1, H):
        for yy in (y - d, y + d):
            if 0 <= yy < H and not cov[yy][x] and not (x in runs and runs[x][0] <= yy <= runs[x][1]):
                return frame[yy][x]
    return frame[y][x]


def legend(frame, blocks, cov):
    out = {}
    for b in blocks:
        c = b["pixels"][0][0]
        pts = [(x, y) for y in range(len(frame)) for x in range(len(frame[0])) if frame[y][x] == c and not cov[y][x]]
        if pts:
            out[b["name"]] = (min(p[0] for p in pts), min(p[1] for p in pts), c)
    return out


def bar_count(frame):
    return sum(1 for v in frame[bar_row(frame)] if v == 3)


def bar_row(frame):
    for y in range(len(frame)):
        if all(v in (2, 3) for v in frame[y]):
            return y
    return len(frame) - 1


def transition_function(state, action, frame):
    state = copy.deepcopy(state)
    H, W = len(frame), len(frame[0])
    cov = [[False] * W for _ in range(H)]
    for o in state:
        if o.get("visible", True):
            for x, y, v in cells(o):
                if 0 <= x < W and 0 <= y < H:
                    cov[y][x] = True
    ring = next(o for o in state if o["type"] == "player")
    beam = next(o for o in state if o["type"] == "arm")
    blocks = [o for o in state if o["type"] == "block"]
    others = [o for o in state if o["type"] not in ("player", "arm", "block")]
    leg = legend(frame, blocks, cov)
    order = [n for n, _ in sorted(leg.items(), key=lambda kv: (kv[1][0], kv[1][1]))]

    b0 = bar_count(frame)
    k = _mem["k"] if _mem["frame"] == frame and _mem["k"] is not None else (0 if b0 == 0 else 3 * b0 + 1)
    click = isinstance(action, dict)
    if not click:
        k += 1
        nr, nbm, nbl = step(ring, beam, blocks, action)
        update_tags(nbl, nbm, order)
    else:
        nr, nbm, nbl = ring, beam, blocks
    after = [nr, nbm] + nbl + others

    runs = rail_runs(frame, cov)
    out = [row[:] for row in frame]
    for y in range(H):
        for x in range(W):
            if cov[y][x]:
                out[y][x] = background(frame, cov, x, y, runs)
    for o in sorted(after, key=lambda o: o["layer"]):
        if o.get("visible", True):
            for x, y, v in cells(o):
                if 0 <= x < W and 0 <= y < H:
                    out[y][x] = v
    for b in nbl:
        if b["name"] in leg:
            lx, ly, c = leg[b["name"]]
            for yy in (ly + 1, ly + 2):
                for xx in (lx + 1, lx + 2):
                    out[yy][xx] = 0 if kind(b) == "collected" else c
    if not click:
        br = bar_row(frame)
        n = max(0, (k - 1) // 3)
        for x in range(W - n, W):
            out[br][x] = 3
    _mem["frame"], _mem["k"] = out, k
    return out
