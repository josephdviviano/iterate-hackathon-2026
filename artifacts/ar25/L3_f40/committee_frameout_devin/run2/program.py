# Mechanics: 21x21 board of 3x3 blocks (bg 9); a horizontal wall axis row (ring 10, dot 0 when selected),
# solid targets (11), pieces (ring 5, centre 0 when selected) and reflections (ring 4, centre 4) = every
# piece block off the axis row mirrored to row 2a-r, clipped to the board, never drawn on piece blocks.
# A1/A2 move the selected axis (A3/A4 no-op) or the selected piece (A1-4, blocked by board/other pieces);
# A5 cycles axis -> pieces by size desc; HUD column 63 gains one 12 per effective action (no-ops free). Unconfirmed: pieces onto axis row.
N, S, BG = 21, 3, 9
WALL, TARGET, PIECE, REFL, HUD_ON, HUD_OFF = 10, 11, 5, 4, 12, 11
RING = [(dx, dy) for dy in range(3) for dx in range(3) if (dx, dy) != (1, 1)]
_memo = {"frame": None, "model": None}


def ring_is(frame, c, r, col):
    return all(frame[3 * r + dy][3 * c + dx] == col for dx, dy in RING)


def centre(frame, c, r):
    return frame[3 * r + 1][3 * c + 1]


def components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c, r = stack.pop()
            comp.add((c, r))
            for n in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def order_pieces(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def split_selected(frame, comp):
    # touching pieces merge; selected blocks show 0 (or a target), unselected show the floor
    sel = {b for b in comp if centre(frame, *b) == 0}
    if not sel or len(sel) == len(comp):
        return [comp]
    grow = True
    while grow:
        grow = False
        for c, r in comp - sel:
            if centre(frame, c, r) == TARGET and any(n in sel for n in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1))):
                sel.add((c, r))
                grow = True
    return [sel] + components(comp - sel)


def parse(frame):
    blocks = [(c, r) for r in range(N) for c in range(N)]
    wall_rows = {}
    for c, r in blocks:
        if ring_is(frame, c, r, WALL):
            wall_rows[r] = wall_rows.get(r, 0) + 1
    axis = max(wall_rows, key=lambda r: wall_rows[r]) if wall_rows else N // 2
    wall_sel = any(ring_is(frame, c, axis, WALL) and centre(frame, c, axis) == 0 for c in range(N))
    targets = set()
    for c, r in blocks:
        if all(frame[3 * r + dy][3 * c + dx] == TARGET for dy in range(3) for dx in range(3)):
            targets.add((c, r))
        elif centre(frame, c, r) == TARGET and (ring_is(frame, c, r, PIECE) or ring_is(frame, c, r, REFL)):
            targets.add((c, r))
    pblocks = [b for b in blocks if ring_is(frame, b[0], b[1], PIECE)]
    comps = components(pblocks)
    if not wall_sel:
        comps = [q for p in comps for q in split_selected(frame, p)]
    pieces = order_pieces(comps)
    sel = "axis" if wall_sel else None
    if not wall_sel:
        for i, p in enumerate(pieces):
            if any(centre(frame, c, r) == 0 for c, r in p):
                sel = i
    return {"axis": axis, "sel": sel, "targets": targets, "pieces": pieces}


def piece_blocked(pieces, i, moved):
    others = set().union(*[p for j, p in enumerate(pieces) if j != i]) if len(pieces) > 1 else set()
    return any(not (0 <= c < N and 0 <= r < N) or (c, r) in others for c, r in moved)


def step(m, action):
    m = {"axis": m["axis"], "sel": m["sel"], "targets": set(m["targets"]),
         "pieces": [set(p) for p in m["pieces"]]}
    aid = action.get("action_id") if isinstance(action, dict) else action
    deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    if aid == 5:
        order = ["axis"] + list(range(len(m["pieces"])))
        cur = order.index(m["sel"]) if m["sel"] in order else 0
        m["sel"] = order[(cur + 1) % len(order)]
    elif aid in deltas:
        dx, dy = deltas[aid]
        if m["sel"] == "axis" or m["sel"] is None:
            if dx == 0 and m["sel"] == "axis" and 0 <= m["axis"] + dy < N:
                m["axis"] += dy
        else:
            i = m["sel"]
            moved = {(c + dx, r + dy) for c, r in m["pieces"][i]}
            if not piece_blocked(m["pieces"], i, moved):
                m["pieces"][i] = moved
    return m


def reflections(m):
    a, occupied = m["axis"], set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    for c, r in occupied:
        d = 2 * a - r
        if r != a and 0 <= d < N and (c, d) not in occupied:
            out.add((c, d))
    return out


def render(m, frame):
    out = [row[:] for row in frame]
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m["pieces"]):
        for b in p:
            owner[b] = i
    for r in range(N):
        for c in range(N):
            # layers bottom-up: (ring colour, centre fill or None for solid)
            layers = []
            if r == m["axis"]:
                layers.append((WALL, 0 if m["sel"] == "axis" else -1))
            if (c, r) in m["targets"]:
                layers.append((TARGET, None))
            if (c, r) in refl:
                layers.append((REFL, REFL))
            if (c, r) in owner:
                layers.append((PIECE, 0 if m["sel"] == owner[(c, r)] else -1))
            ring = layers[-1][0] if layers else BG
            mid = BG
            if any(f is None for _, f in layers):
                mid = TARGET
            else:
                for _, f in reversed(layers):
                    if f >= 0:
                        mid = f
                        break
            for dy in range(3):
                for dx in range(3):
                    out[3 * r + dy][3 * c + dx] = mid if (dx, dy) == (1, 1) else ring
    return out


def tick_hud(out):
    for y in range(len(out)):
        if out[y][63] == HUD_OFF:
            out[y][63] = HUD_ON
            break


def transition_function(state, action, frame):
    if _memo["frame"] is not None and frame == _memo["frame"]:
        model = _memo["model"]
    else:
        model = parse(frame)
    new = step(model, action)
    out = render(new, frame)
    if new != model:
        tick_hud(out)
    _memo["frame"], _memo["model"] = [row[:] for row in out], new
    return out
