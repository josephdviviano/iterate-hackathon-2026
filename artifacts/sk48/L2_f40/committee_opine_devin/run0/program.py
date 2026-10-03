# Mechanics: ring (player) + beam (arm, x=ring.x+5, h=2, w=1+6k) move together on A1/A2 (y-/+6), carrying
# blocks that overlap the beam and pushing blocks they hit (chain); any block leaving the colour-4 play rect cancels.
# A4/A3 extend/retract the beam by 6 (w in [1, play_right-x+1]); hooked next/collected block (x == tip-4) drags all
# on-beam blocks, unhooked extend pushes swept blocks; a failing group leaves the beam to change alone. Clicks no-op.
# HUD: bar row gains a 3 from the right every 3rd non-click action (hidden counter n, continuity-gated); legend
# swatch centre 2x2 = 0 while that block is collected. Hypothesis: arm A4 no_change = beam already at max width.
import copy

_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def hud_row(frame):
    for y, row in enumerate(frame):
        if all(v in (2, 3) for v in row):
            return y
    return len(frame)


def play_rect(frame, bar):
    cells = [(x, y) for y in range(bar) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def block_ok(b, pr):
    return b["x"] >= pr[0] and b["y"] >= pr[1] and b["x"] + b["w"] - 1 <= pr[2] and b["y"] + b["h"] - 1 <= pr[3]


def push_chain(blocks, movers, solids, dx, dy):
    """Move movers by (dx,dy); blocks hit by moved blocks or by solids are pushed too. Returns moved ids."""
    moved = set(movers)
    while True:
        occ = list(solids) + [(blocks[i]["x"] + dx, blocks[i]["y"] + dy, blocks[i]["w"], blocks[i]["h"]) for i in moved]
        new = {i for i, b in enumerate(blocks) if i not in moved and any(overlap(rect(b), r) for r in occ)}
        if not new:
            return moved
        moved |= new


def apply_move(blocks, moved, dx, dy, pr):
    trial = copy.deepcopy(blocks)
    for i in moved:
        trial[i]["x"] += dx
        trial[i]["y"] += dy
    return trial if all(block_ok(trial[i], pr) for i in moved) else None


def status(b):
    for t in ("collected", "next", "todo"):
        if t in b.get("tags", []):
            return t
    return "todo"


def set_status(b, s):
    b["tags"] = [t for t in b["tags"] if t not in ("collected", "next", "todo")] + [s]


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    return [i for i, b in enumerate(blocks)
            if status(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == tip - 4]


def update_tags(blocks, beam, order):
    hk = hooked(blocks, beam)
    for i, b in enumerate(blocks):
        if status(b) == "collected" and not on_beam(b, beam):
            for o in blocks:
                if status(o) == "next":
                    set_status(o, "todo")
            set_status(b, "next")
    for i in hk:
        b = blocks[i]
        others = [o for j, o in enumerate(blocks) if j != i and on_beam(o, beam)]
        if status(b) == "next" and all(status(o) == "collected" for o in others):
            set_status(b, "collected")
            todo = [o for o in blocks if status(o) == "todo"]
            todo.sort(key=lambda o: order.get(o["pixels"][0][0], 99))
            if todo:
                set_status(todo[0], "next")


def step(objs, action, pr, order):
    ring = next(o for o in objs if o["type"] == "player")
    beam = next(o for o in objs if o["type"] == "arm")
    blocks = [o for o in objs if o["type"] == "block"]
    if action in (1, 2):
        dy = -6 if action == 1 else 6
        if ring["y"] + dy < pr[1] or ring["y"] + dy + ring["h"] - 1 > pr[3]:
            return objs
        carried = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
        solids = [(ring["x"], ring["y"] + dy, ring["w"], ring["h"]), (beam["x"], beam["y"] + dy, beam["w"], beam["h"])]
        trial = apply_move(blocks, push_chain(blocks, carried, solids, 0, dy), 0, dy, pr)
        if trial is None:
            return objs
        ring["y"] += dy
        beam["y"] += dy
        blocks = trial
    elif action in (3, 4):
        dw = 6 if action == 4 else -6
        nw = beam["w"] + dw
        if nw < 1 or beam["x"] + nw - 1 > pr[2]:
            return objs
        hk = hooked(blocks, beam)
        if hk:
            movers = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
            trial = apply_move(blocks, push_chain(blocks, movers, [], dw, 0), dw, 0, pr)
        elif dw > 0:
            swept = [(beam["x"] + beam["w"], beam["y"], dw, beam["h"])]
            trial = apply_move(blocks, push_chain(blocks, set(), swept, dw, 0), dw, 0, pr)
        else:
            trial = None
        beam["w"] = nw
        if trial is not None:
            blocks = trial
    else:
        return objs
    update_tags(blocks, beam, order)
    return [ring, beam] + blocks


def background(x, y, pr, ring):
    if pr[0] <= x <= pr[2] and pr[1] <= y <= pr[3]:
        return 4
    if x in (ring["x"] + 2, ring["x"] + 3) and pr[1] + 2 <= y <= pr[3] - 2:
        return 2 if (y - pr[1] - 2) % 6 < 2 else 3
    return 5


def draw(out, o):
    for dy in range(o["h"]):
        for dx in range(o["w"]):
            x, y = o["x"] + dx, o["y"] + dy
            if not (0 <= x < 64 and 0 <= y < 64):
                continue
            if o["type"] == "arm":
                v = 2 if (dx % 3 == 1 if dy == 0 else dx % 3 == 0) else 1
            elif o["type"] == "block":
                v = o["pixels"][0][0]
            else:
                v = o["pixels"][dy][dx]
            out[y][x] = v


def legend(frame, bar, colour):
    cells = [(x, y) for y in range(bar + 1, 64) for x in range(64) if frame[y][x] == colour]
    if not cells:
        return None
    return min(c[0] for c in cells), min(c[1] for c in cells)


def transition_function(state, action, frame):
    bar = hud_row(frame)
    pr = play_rect(frame, bar)
    objs = copy.deepcopy(state)
    blocks = [o for o in objs if o["type"] == "block"]
    pos = {b["pixels"][0][0]: legend(frame, bar, b["pixels"][0][0]) for b in blocks}
    order = {c: rank for rank, c in enumerate(sorted((c for c in pos if pos[c]), key=lambda c: pos[c]))}
    ring0 = next(o for o in objs if o["type"] == "player")
    after = step(copy.deepcopy(objs), action, pr, order)
    out = [list(r) for r in frame]
    for o in objs:
        for dy in range(o["h"]):
            for dx in range(o["w"]):
                x, y = o["x"] + dx, o["y"] + dy
                if 0 <= x < 64 and 0 <= y < bar:
                    out[y][x] = background(x, y, pr, ring0)
    for o in sorted(after, key=lambda o: o["layer"]):
        draw(out, o)
    for b in after:
        if b["type"] == "block" and pos.get(b["pixels"][0][0]):
            lx, ly = pos[b["pixels"][0][0]]
            v = 0 if status(b) == "collected" else b["pixels"][0][0]
            for yy in (ly + 1, ly + 2):
                for xx in (lx + 1, lx + 2):
                    out[yy][xx] = v
    if bar < 64:
        if _mem["frame"] == frame:
            n = _mem["n"]
        else:
            k = sum(1 for v in frame[bar] if v == 3)
            n = 3 * k + 1 if k else 0
        if not isinstance(action, dict):
            n += 1
        filled = max(0, (n - 1) // 3)
        for i in range(filled):
            out[bar][63 - i] = 3
        _mem["n"] = n
    _mem["frame"] = out
    return out
