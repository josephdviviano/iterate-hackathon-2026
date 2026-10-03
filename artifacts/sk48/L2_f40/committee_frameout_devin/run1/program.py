# Mechanics: crane ring (player) slides on the left rail carrying a beam (arm) that extends/retracts by 6 (A4/A3);
# A1/A2 move ring+beam 6 rows, carrying blocks on the beam and pushing blocks the new rects hit (cancel if any block leaves floor 4).
# A hooked next/collected block (right edge at beam tip-1) drags on-beam blocks; else extension pushes next/collected blocks in the swept
# columns; a blocked chain lets the beam change alone. Tags: hooked next collects, off-beam collected reverts. HUD row: every 3rd non-click
# action turns a 2 to 3 from the right (phase hidden, continuity-gated). Hypotheses: rail spans field rows inset 2; clicks are no-ops.
FLOOR, BG = 4, 5
_mem = {"frame": None, "count": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def infer_bg(frame, objs, ring):
    covered = set()
    for o in objs:
        covered.update(cells(rect(o)))
    bg = [row[:] for row in frame]
    col = ring["x"] + ring["w"]
    top = bot = ring["y"] + 2
    while top > 0 and frame[top - 1][col] != BG:
        top -= 1
    while bot < 63 and frame[bot + 1][col] != BG:
        bot += 1
    for (x, y) in covered:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        cands = []
        for step in (-6, 6):
            yy = y + step
            while 0 <= yy < 64 and (x, yy) in covered:
                yy += step
            if 0 <= yy < 64:
                cands.append(frame[yy][x])
        other = [c for c in cands if c != BG]
        if not other:
            bg[y][x] = BG
        elif other[0] == FLOOR or top + 2 <= y <= bot - 2:
            bg[y][x] = other[0]
        else:
            bg[y][x] = BG
    return bg


def floor_ok(bg, r):
    return all(0 <= x < 64 and 0 <= y < 64 and bg[y][x] == FLOOR for (x, y) in cells(r))


def push_chain(blocks, movers, dx, dy, solids, bg, ring_r):
    moving = set(movers)
    while True:
        moved = [(b["x"] + dx, b["y"] + dy, b["w"], b["h"]) for b in blocks if b["name"] in moving]
        hit = [b["name"] for b in blocks if b["name"] not in moving and
               any(overlap(rect(b), r) for r in moved + solids)]
        if not hit:
            break
        moving.update(hit)
    for b in blocks:
        if b["name"] in moving:
            r = (b["x"] + dx, b["y"] + dy, b["w"], b["h"])
            if not floor_ok(bg, r) or overlap(r, ring_r):
                return None
    return {b["name"]: (b["x"] + dx, b["y"] + dy) for b in blocks if b["name"] in moving}


def status(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return "todo"


def tip(beam):
    return beam["x"] + beam["w"] - 1


def on_beam(b, beam):
    return overlap(rect(b), rect(beam))


def hooked(b, beam):
    return status(b) in ("next", "collected") and on_beam(b, beam) and b["x"] == tip(beam) - 4


def step(ring, beam, blocks, action, bg):
    ring_r = rect(ring)
    if action in (1, 2):
        dy = -6 if action == 1 else 6
        nring = dict(ring, y=ring["y"] + dy)
        nbeam = dict(beam, y=beam["y"] + dy)
        if not floor_ok(bg, (ring["x"] + ring["w"], nbeam["y"], 1, nbeam["h"])):
            return
        carried = [b["name"] for b in blocks if on_beam(b, beam)]
        res = push_chain(blocks, carried, 0, dy, [rect(nring), rect(nbeam)], bg, ring_r)
        if res is None:
            return
        ring["y"], beam["y"] = nring["y"], nbeam["y"]
        apply(blocks, res)
    elif action in (3, 4):
        dx = 6 if action == 4 else -6
        nw = beam["w"] + dx
        if action == 4 and not floor_ok(bg, (tip(beam) + dx, beam["y"], 1, beam["h"])):
            return
        if nw < 1:
            return
        hook = [b for b in blocks if hooked(b, beam)]
        if hook:
            movers = [b["name"] for b in blocks if on_beam(b, beam)]
        elif action == 4:
            swept = (tip(beam) + 1, beam["y"], dx, beam["h"])
            movers = [b["name"] for b in blocks if status(b) != "todo" and overlap(rect(b), swept)]
        else:
            movers = []
        if movers:
            res = push_chain(blocks, movers, dx, 0, [], bg, ring_r)
            if res is not None:
                apply(blocks, res)
        beam["w"] = nw


def apply(blocks, res):
    for b in blocks:
        if b["name"] in res:
            b["x"], b["y"] = res[b["name"]]


def set_status(b, s):
    b["tags"] = [t for t in b["tags"] if t not in ("collected", "next", "todo")] + [s]


def update_tags(blocks, beam):
    nxt = [b for b in blocks if status(b) == "next"]
    hooked_any = any(hooked(b, beam) for b in blocks)
    for b in nxt:
        others = [o for o in blocks if o is not b and on_beam(o, beam)]
        if hooked(b, beam) and all(status(o) == "collected" for o in others):
            set_status(b, "collected")
            rest = [o for o in blocks if status(o) != "collected"]
            if rest:
                set_status(max(rest, key=lambda o: o["x"]), "next")
            return
    if not hooked_any:
        off = [b for b in blocks if status(b) == "collected" and not on_beam(b, beam)]
        if off:
            for b in blocks:
                if status(b) == "next":
                    set_status(b, "todo")
            set_status(max(off, key=lambda o: o["x"]), "next")


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def draw(out, o, pixels):
    for j, row in enumerate(pixels):
        for i, v in enumerate(row):
            x, y = o["x"] + i, o["y"] + j
            if 0 <= x < 64 and 0 <= y < 64 and v >= 0:
                out[y][x] = v


def hud_row(frame):
    for y in range(64):
        if all(v in (2, 3) for v in frame[y]) and 3 in frame[y] or all(v == 2 for v in frame[y]):
            return y
    return None


def transition_function(state, action, frame):
    objs = [dict(o, tags=list(o.get("tags", []))) for o in state]
    ring = next(o for o in objs if o["type"] == "player")
    beam = next(o for o in objs if o["type"] == "arm")
    blocks = [o for o in objs if o["type"] == "block"]
    old = [dict(o) for o in objs]
    bg = infer_bg(frame, objs, ring)
    hy = hud_row(frame)
    z = sum(1 for v in frame[hy] if v == 3) if hy is not None else 0
    count = _mem["count"] if _mem["frame"] == frame else (3 * z + 1 if z else 0)
    if isinstance(action, int):
        step(ring, beam, blocks, action, bg)
        update_tags(blocks, beam)
        count += 1
    out = [row[:] for row in frame]
    for o in old:
        for (x, y) in cells(rect(o)):
            if 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = bg[y][x]
    draw(out, ring, ring["pixels"])
    draw(out, beam, beam_pixels(beam["w"]))
    for b in sorted(blocks, key=lambda o: o["layer"]):
        draw(out, b, b["pixels"])
    if hy is not None:
        marks = max(0, (count - 1) // 3)
        for i in range(64):
            out[hy][63 - i] = 3 if i < marks else 2
        for b in blocks:
            c = b["pixels"][0][0]
            pts = [(x, y) for y in range(hy + 1, 64) for x in range(64) if frame[y][x] == c]
            if not pts:
                continue
            x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
            y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
            for y in range((y0 + y1) // 2, (y0 + y1 + 1) // 2 + 1):
                for x in range((x0 + x1) // 2, (x0 + x1 + 1) // 2 + 1):
                    out[y][x] = 0 if status(b) == "collected" else c
    _mem["frame"], _mem["count"] = [row[:] for row in out], count
    return out
