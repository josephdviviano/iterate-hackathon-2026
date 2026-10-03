# Mechanics: 21x21 board of 3x3 blocks (bg 9); a horizontal axis band (ring 10, centre 0 when selected),
# solid targets 11, pieces ring 5 (centre 0 when selected), reflections ring 4 / centre 4 at row 2A-r.
# A1/A2 move the selection a block up/down (axis unblocked; pieces blocked by edge/other pieces),
# A3/A4 move a selected piece left/right (axis: no-op), A5 cycles axis -> pieces by size desc -> axis.
# HUD col 63 gains one 12 per effective action. Hidden state: last model reused only on exact frame continuity.
N = 21
BG, AXIS, TARGET, REFL, PIECE, SEL, HUD_ON = 9, 10, 11, 4, 5, 0, 12

_memo = {"frame": None, "model": None}


def blk(frame, bx, by, dx=0, dy=0):
    return frame[3 * by + dy][3 * bx + dx]


def has_target(frame, bx, by):
    return any(blk(frame, bx, by, i, j) == TARGET for i in range(3) for j in range(3))


def comps(cells):
    cells, out = set(cells), []
    while cells:
        seed = cells.pop()
        comp, stack = {seed}, [seed]
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(frame):
    targets, piece_cells, axis_votes = set(), {}, {}
    for by in range(N):
        for bx in range(N):
            ring, centre = blk(frame, bx, by), blk(frame, bx, by, 1, 1)
            if has_target(frame, bx, by):
                targets.add((bx, by))
            if ring == AXIS:
                axis_votes[by] = axis_votes.get(by, 0) + 1
            if ring == PIECE:
                piece_cells[(bx, by)] = centre
    axis = max(axis_votes, key=lambda r: axis_votes[r]) if axis_votes else 0
    axis_sel = any(blk(frame, bx, axis) == AXIS and blk(frame, bx, axis, 1, 1) == SEL for bx in range(N))
    pieces, sel = [], None
    for comp in comps(piece_cells):
        dots = {c for c in comp if piece_cells[c] == SEL}
        grown = set()
        if dots and not axis_sel:
            grown, stack = set(dots), list(dots)
            while stack:
                x, y = stack.pop()
                for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if n in comp and n not in grown and piece_cells[n] != BG:
                        grown.add(n)
                        stack.append(n)
            sel = len(pieces)
            pieces.append(grown)
        pieces.extend(comps(comp - grown))
    return {"axis": axis, "sel": "axis" if axis_sel or sel is None else sel,
            "pieces": pieces, "targets": targets}


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min((y, x) for x, y in pieces[i])))


def piece_can_move(model, i, dx, dy):
    others = set().union(*[p for j, p in enumerate(model["pieces"]) if j != i]) if len(model["pieces"]) > 1 else set()
    for x, y in model["pieces"][i]:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < N and 0 <= ny < N) or (nx, ny) in others:
            return False
    return True


def step(model, action):
    sel = model["sel"]
    if action == 5:
        order = cycle_order(model["pieces"])
        if sel == "axis":
            model["sel"] = order[0] if order else "axis"
        else:
            k = order.index(sel)
            model["sel"] = order[k + 1] if k + 1 < len(order) else "axis"
        return model["sel"] != sel
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    if action not in moves:
        return False
    dx, dy = moves[action]
    if sel == "axis":
        if dx or not 0 <= model["axis"] + dy < N:
            return False
        model["axis"] += dy
        return True
    if not piece_can_move(model, sel, dx, dy):
        return False
    model["pieces"][sel] = {(x + dx, y + dy) for x, y in model["pieces"][sel]}
    return True


def render(model, frame):
    out = [row[:] for row in frame]
    occupied = {}
    for i, p in enumerate(model["pieces"]):
        for c in p:
            occupied[c] = i
    axis = model["axis"]
    refl = set()
    for c in occupied:
        x, y = c
        r = (x, 2 * axis - y)
        if y != axis and 0 <= r[1] < N and r not in occupied:
            refl.add(r)
    axis_sel = model["sel"] == "axis"
    for by in range(N):
        for bx in range(N):
            c = (bx, by)
            layers = []  # top-down (ring colour, centre solid?, centre fill)
            if c in occupied:
                layers.append((PIECE, None, SEL if occupied[c] == model["sel"] else None))
            if c in refl:
                layers.append((REFL, None, REFL))
            if c in model["targets"]:
                layers.append((TARGET, TARGET, None))
            if by == axis:
                layers.append((AXIS, SEL if axis_sel else None, None))
            ring = layers[0][0] if layers else BG
            centre = next((s for _, s, _ in layers if s is not None), None)
            if centre is None:
                centre = next((f for _, _, f in layers if f is not None), BG)
            for j in range(3):
                for i in range(3):
                    out[3 * by + j][3 * bx + i] = centre if (i, j) == (1, 1) else ring
    return out


def transition_function(state, action, frame):
    if _memo["frame"] is not None and frame == _memo["frame"]:
        model = _memo["model"]
        model = {"axis": model["axis"], "sel": model["sel"], "targets": set(model["targets"]),
                 "pieces": [set(p) for p in model["pieces"]]}
    else:
        model = parse(frame)
    act = action if isinstance(action, int) else action.get("action_id")
    changed = step(model, act)
    out = render(model, frame)
    if changed:
        for y in range(63):
            if out[y][63] != HUD_ON:
                out[y][63] = HUD_ON
                break
    _memo["frame"], _memo["model"] = out, model
    return out
