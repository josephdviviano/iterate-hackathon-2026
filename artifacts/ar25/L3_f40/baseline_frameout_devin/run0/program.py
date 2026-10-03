# Mechanics: 21x21 board of 3x3 blocks; layers axis band (10, centre dots 0 if selected) < targets
# (solid 11) < reflections (ring 4, fill 4) < pieces (ring 5, fill 0 if selected, else transparent).
# A1-A4 move selection 1 block (axis rows only, not onto piece rows; pieces stay in board, off axis
# row/others); A5 cycles axis -> pieces largest first -> axis; reflection row 2a-r, clipped, not under
# pieces. HUD: effective actions turn first 11 of col 63 into 12. Unconfirmed: piece->piece A5 order.
N, BG = 21, 9
AXIS, TARGET, REFL, PIECE, DOT, TICK = 10, 11, 4, 5, 0, 12
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memory = {"frame": None, "model": None}


def block_cells(f, r, c):
    ring = [f[3 * r + dy][3 * c + dx] for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]
    return set(ring), f[3 * r + 1][3 * c + 1]


def components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                if q in cells:
                    cells.discard(q)
                    stack.append(q)
        comps.append(comp)
    return comps


def parse(f):
    targets, axis_row, axis_sel = set(), None, False
    sel_cells, unsel_cells, amb_cells = set(), set(), set()
    for r in range(N):
        for c in range(N):
            ring, centre = block_cells(f, r, c)
            if len(ring) != 1:
                continue
            ring = ring.pop()
            if ring == TARGET or (ring in (PIECE, REFL) and centre == TARGET):
                targets.add((r, c))
            if ring == AXIS:
                axis_row = r
                axis_sel = axis_sel or centre == DOT
            if ring == PIECE:
                (sel_cells if centre == DOT else unsel_cells if centre == BG else amb_cells).add((r, c))
    pieces = []
    for comp in components(sel_cells | unsel_cells | amb_cells):
        sel = comp & sel_cells
        if not sel:
            pieces.append((comp, False))
            continue
        grown = set()
        for sub in components(sel | (comp & amb_cells)):
            if sub & sel:
                grown |= sub
        pieces.append((grown, True))
        for rest in components(comp - grown):
            pieces.append((rest, False))
    selection = "axis" if axis_sel else None
    plist = [p for p, _ in pieces]
    for i, (_, s) in enumerate(pieces):
        if s and selection is None:
            selection = i
    return {"axis": axis_row, "targets": targets, "pieces": plist, "sel": selection}


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(model, action):
    pieces = [set(p) for p in model["pieces"]]
    axis, sel = model["axis"], model["sel"]
    if action == 5:
        order = cycle_order(pieces)
        if sel == "axis" or sel is None:
            sel = order[0] if order else "axis"
        else:
            k = order.index(sel)
            sel = order[k + 1] if k + 1 < len(order) else "axis"
    elif action in MOVES:
        dr, dc = MOVES[action]
        if sel == "axis" and axis is not None and dc == 0:
            new = axis + dr
            piece_rows = {r for p in pieces for r, _ in p}
            if 0 <= new < N and new not in piece_rows:
                axis = new
        elif isinstance(sel, int):
            moved = {(r + dr, c + dc) for r, c in pieces[sel]}
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel])
            if all(0 <= r < N and 0 <= c < N and r != axis and (r, c) not in others for r, c in moved):
                pieces[sel] = moved
    return {"axis": axis, "targets": set(model["targets"]), "pieces": pieces, "sel": sel}


def reflections(model):
    occupied = set().union(*model["pieces"]) if model["pieces"] else set()
    a, out = model["axis"], set()
    if a is None:
        return out
    for r, c in occupied:
        d = 2 * a - r
        if r != a and 0 <= d < N and (d, c) not in occupied:
            out.add((d, c))
    return out


def paint_block(g, r, c, ring, centre):
    for dy in range(3):
        for dx in range(3):
            if (dy, dx) == (1, 1):
                if centre is not None:
                    g[3 * r + 1][3 * c + 1] = centre
            else:
                g[3 * r + dy][3 * c + dx] = ring


def render(model, frame):
    g = [row[:] for row in frame]
    for y in range(3 * N):
        for x in range(3 * N):
            g[y][x] = BG
    a, targets = model["axis"], model["targets"]
    if a is not None:
        for c in range(N):
            paint_block(g, a, c, AXIS, DOT if model["sel"] == "axis" else BG)
    for r, c in targets:
        paint_block(g, r, c, TARGET, TARGET)
    for r, c in reflections(model):
        paint_block(g, r, c, REFL, TARGET if (r, c) in targets else REFL)
    for i, p in enumerate(model["pieces"]):
        for r, c in p:
            if (r, c) in targets:
                centre = TARGET
            else:
                centre = DOT if model["sel"] == i else BG
            paint_block(g, r, c, PIECE, centre)
    return g


def tick_hud(g):
    x = len(g[0]) - 1
    for y in range(len(g)):
        if g[y][x] == TARGET:
            g[y][x] = TICK
            break


def changed(old, new):
    return (old["axis"], old["sel"], sorted(map(sorted, old["pieces"]))) != \
        (new["axis"], new["sel"], sorted(map(sorted, new["pieces"])))


def transition_function(state, action, frame):
    act = action.get("action_id") if isinstance(action, dict) else action
    if _memory["frame"] is not None and frame == _memory["frame"]:
        model = _memory["model"]
    else:
        model = parse(frame)
    new = step(model, act)
    out = render(new, frame)
    if changed(model, new):
        tick_hud(out)
    _memory["frame"], _memory["model"] = [row[:] for row in out], new
    return out
