# Mechanics: player 5x5 moves 6 (A1-4) inside field rows>=8; wall px block except a box holding the plate centre;
# first input of a level (move count 0) is a no-op. Plate covered by player/ghost -> arm+key retract 6 toward the line.
# A5 off-respawn toggles ghost mode: player -> respawn, ghost HUD icon at (1,1), old huds x+4; ghost = trail 2 moves
# behind (trail restarts [start, respawn]); A5 with ghost clears it. Counter w = 64 - ceil(n/2), n = calls (hidden).
# Unconfirmed: invisible cell vs first-input no-op at t0; respawn = first reached cell; stateless parity/ghost trail.
import json

FIELD_TOP, STEP, SIZE = 8, 6, 5
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MEM = {"last": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def body_pixels(color):
    p = [[color] * SIZE for _ in range(SIZE)]
    p[2][2] = -1
    return p


def obj(typ, tags, x, y, pixels, layer=0):
    return {"type": typ, "tags": list(tags), "x": x, "y": y, "w": len(pixels[0]), "h": len(pixels),
            "layer": layer, "visible": True, "pixels": pixels, "name": ""}


def wall_cells(o):
    return {(o["x"] + j, o["y"] + i) for i, r in enumerate(o["pixels"]) for j, v in enumerate(r) if v >= 0}


def in_box(c, b):
    return b[0] <= c[0] < b[0] + SIZE and b[1] <= c[1] < b[1] + SIZE


# ---- wall (plate -> line -> arm -> key) ----
def line_col(cells):
    cols = {}
    for x, _ in cells:
        cols[x] = cols.get(x, 0) + 1
    return max(cols, key=lambda c: (cols[c], -c))


def plate_visible(cells, lc):
    top = min(y for _, y in cells)
    return {x for x, y in cells if y == top} == {lc - 1, lc, lc + 1}


def retract(cells, lc, sign):
    """sign=+1: shift side pixels toward the line by STEP; -1: extend back and refill the arm row."""
    out = {c for c in cells if c[0] == lc}
    for x, y in cells:
        if x == lc:
            continue
        d = 1 if x < lc else -1
        nx = x + sign * d * STEP
        if (nx - lc) * d < 0:
            out.add((nx, y))
    if sign < 0:
        arm = max(y for x, y in cells if x == lc)
        xs = [x for x, y in out if y == arm]
        for x in range(min(xs), max(xs) + 1):
            out.add((x, arm))
    return out


def parse_wall(cells, bodies):
    """Return (extended wall cells, plate centre)."""
    lc = line_col(cells)
    if plate_visible(cells, lc):
        top = min(y for _, y in cells)
        return set(cells), (lc, top + 1)
    cov = [b for b in bodies if b[0] <= lc < b[0] + SIZE]
    b = min(cov, key=lambda b: b[1]) if cov else None
    cy = b[1] + 2 if b else min(y for _, y in cells) - 2
    full = retract(cells, lc, -1)
    full |= {(lc + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    top_line = min(y for x, y in cells if x == lc)
    full |= {(lc, y) for y in range(cy + 2, top_line)}
    return full, (lc, cy)


def render_wall(full, plate, bodies):
    lc = plate[0]
    cells = retract(full, lc, 1) if any(in_box(plate, b) for b in bodies) else set(full)
    cells = {c for c in cells if not any(in_box(c, b) for b in bodies)}
    if not cells:
        return None
    x0, y0 = min(x for x, _ in cells), min(y for _, y in cells)
    x1, y1 = max(x for x, _ in cells), max(y for _, y in cells)
    px = [[8 if (x, y) in cells else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
    return obj("wall", ["wall", "obstacle"], x0, y0, px)


# ---- player movement guard ----
def blocked(pos, wall_full, plate):
    x, y = pos
    if x < 0 or y < FIELD_TOP or x + SIZE > 64 or y + SIZE > 63:
        return True
    if in_box(plate, pos):
        return False
    return any(in_box(c, pos) for c in wall_full)


def fresh_mem(state, by):
    w = by["counter"][0]["w"]
    p = by["player"][0]
    return {"n": 0 if w == 64 else 2 * (64 - w) - 1, "start": (p["x"], p["y"]),
            "respawn": None, "trail": None}


def transition_function(state, action):
    by = {}
    for o in state:
        by.setdefault(o["type"], []).append(o)
    mem = MEM.get("mem") if MEM["last"] == canon(state) else None
    if mem is None:
        mem = fresh_mem(state, by)
    mem = dict(mem)
    aid = action["action_id"] if isinstance(action, dict) else action

    player = by["player"][0]
    pos = (player["x"], player["y"])
    ghost = by.get("ghost", [None])[0]
    gpos = (ghost["x"], ghost["y"]) if ghost else None
    huds = sorted(by["hud"], key=lambda o: o["name"])
    ghost_hud = [h for h in huds if h["pixels"][0][0] == 2]
    base_huds = [h for h in huds if h["pixels"][0][0] != 2]
    mode = bool(ghost_hud)
    bodies = [b for b in (pos, gpos) if b]
    full, plate = parse_wall(wall_cells(by["wall"][0]), bodies)
    trail = list(mem["trail"]) if mem["trail"] is not None else ([gpos, pos] if gpos else None)

    if aid in MOVES and mem["n"] > 0:
        dx, dy = MOVES[aid]
        np_ = (pos[0] + dx * STEP, pos[1] + dy * STEP)
        if not blocked(np_, full, plate):
            pos = np_
            if mem["respawn"] is None:
                mem["respawn"] = pos
            if mode and trail is not None:
                trail.append(pos)
                if len(trail) >= 3:
                    gpos = trail[-3]
    elif aid == 5:
        respawn = mem["respawn"] or pos
        if pos != respawn:
            if mode:
                mode, gpos, trail = False, None, None
            else:
                mode, trail = True, [mem["start"], respawn]
            pos = respawn
    mem["trail"] = trail if mode else None
    mem["n"] += 1

    shift = 4 if mode else 0
    out = [dict(h, x=1 + shift + (h["x"] - (5 if ghost_hud else 1))) for h in base_huds]
    out = [obj("hud", ["hud"], o["x"], o["y"], o["pixels"]) for o in out]
    out.append(obj("player", ["player", "controllable"], pos[0], pos[1], body_pixels(9), 1))
    for t in ("gate", "target"):
        out += [dict(o) for o in by.get(t, [])]
    w = 64 - (mem["n"] + 1) // 2
    out.append(obj("counter", ["hud", "move_counter"], 0, 63, [[9] * w]))
    if mode:
        out.append(obj("hud", ["hud"], 1, 1, [[2, 2, 2], [2, -1, 2], [2, 2, 2]]))
        if gpos:
            out.append(obj("ghost", ["ghost", "snake_tail", "follower"], gpos[0], gpos[1], body_pixels(2), 1))
    wall = render_wall(full, plate, [b for b in (pos, gpos) if b])
    if wall:
        out.append(wall)
    for i, o in enumerate(out):
        o["name"] = "%s_%03d" % (o["type"], i)
    MEM["last"], MEM["mem"] = canon(out), mem
    return out
