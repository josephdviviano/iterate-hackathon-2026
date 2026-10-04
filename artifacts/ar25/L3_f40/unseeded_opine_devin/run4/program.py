# Mechanics: 21x21 grid of 3x3 blocks; axis = full-width wall row (ring 10, centre dot 0 when selected);
# pieces = holed 5-ring blocks (centre 0 when selected), targets = solid 11 blocks, reflections = 4 blocks
# at row 2A-r of every off-axis piece block (skipped off-board / on piece blocks). A1/A2 move selection a row,
# A3/A4 a column (pieces only; blocked by board edge / other pieces); A5 cycles axis -> pieces by size -> axis.
# HUD col 63 turns one 11->12 per effective action. Unconfirmed: axis/piece row-sharing rules beyond observed.
N = 21
BG, WALL, TGT, REFL, PIECE, DOT, HUD_ON = 9, 10, 11, 4, 5, 0, 12
_memo = {"frame": None, "model": None}


def block(frame, bx, by):
    return [frame[3 * by + j][3 * bx + i] for j in range(3) for i in range(3)]


def comps(cells):
    cells, out = set(cells), []
    while cells:
        stack = [cells.pop()]
        comp = set(stack)
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(frame):
    walls, targets, pieces = {}, set(), {}
    for by in range(N):
        for bx in range(N):
            px = block(frame, bx, by)
            corner, centre = px[0], px[4]
            if TGT in px:
                targets.add((bx, by))
            if corner == WALL:
                walls[(bx, by)] = centre
            elif corner == PIECE:
                pieces[(bx, by)] = centre
    rows = {}
    for (bx, by) in walls:
        rows[by] = rows.get(by, 0) + 1
    axis = max(rows, key=rows.get) if rows else 0
    axis_sel = any(c == DOT for c in walls.values())
    plist, sel = [], None
    for comp in comps(pieces):
        dots = {b for b in comp if pieces[b] == DOT} if not axis_sel else set()
        if not dots:
            plist.append(comp)
            continue
        grown, stack = set(dots), list(dots)
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in comp and n not in grown and pieces[n] != BG:
                    grown.add(n)
                    stack.append(n)
        sel = len(plist)
        plist.append(grown)
        plist.extend(comps(comp - grown))
    return {"axis": axis, "sel": None if axis_sel else sel, "pieces": plist, "targets": targets}


def order(model):
    idx = list(range(len(model["pieces"])))
    return sorted(idx, key=lambda i: (-len(model["pieces"][i]), min(model["pieces"][i])))


def step(model, action):
    pieces = [set(p) for p in model["pieces"]]
    axis, sel = model["axis"], model["sel"]
    new = {"axis": axis, "sel": sel, "pieces": pieces, "targets": model["targets"]}
    if action == 5:
        seq = [None] + order(model)
        new["sel"] = seq[(seq.index(sel) + 1) % len(seq)] if sel in seq else None
        return new, new["sel"] != sel
    d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action)
    if d is None:
        return new, False
    dx, dy = d
    if sel is None:
        if dx or not 0 <= axis + dy < N:
            return new, False
        new["axis"] = axis + dy
        return new, True
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel])
    moved = {(x + dx, y + dy) for x, y in pieces[sel]}
    if any(not (0 <= x < N and 0 <= y < N) or (x, y) in others for x, y in moved):
        return new, False
    pieces[sel] = moved
    return new, True


def reflections(model):
    a = model["axis"]
    occ = set().union(*model["pieces"]) if model["pieces"] else set()
    out = set()
    for x, y in occ:
        r = 2 * a - y
        if y != a and 0 <= r < N and (x, r) not in occ:
            out.add((x, r))
    return out


def render(model, frame):
    out = [list(row) for row in frame]
    refl = reflections(model)
    owner = {}
    for i, p in enumerate(model["pieces"]):
        for b in p:
            owner[b] = i
    a, axis_sel = model["axis"], model["sel"] is None
    for by in range(N):
        for bx in range(N):
            b = (bx, by)
            # layers top-down: (ring colour, centre solid?, centre fill)
            layers = []
            if b in owner:
                layers.append((PIECE, None, DOT if owner[b] == model["sel"] else None))
            if b in refl:
                layers.append((REFL, None, REFL))
            if b in model["targets"]:
                layers.append((TGT, TGT, None))
            if by == a:
                layers.append((WALL, DOT if axis_sel else None, None))
            ring = layers[0][0] if layers else BG
            centre = next((l[1] for l in layers if l[1] is not None), None)
            if centre is None:
                centre = next((l[2] for l in layers if l[2] is not None), BG)
            for j in range(3):
                for i in range(3):
                    out[3 * by + j][3 * bx + i] = centre if (i, j) == (1, 1) else ring
    return out


def tick_hud(out):
    for y in range(63):
        if out[y][63] != HUD_ON:
            out[y][63] = HUD_ON
            return


def transition_function(state, action, frame):
    aid = action["action_id"] if isinstance(action, dict) else action
    if _memo["frame"] == frame and _memo["model"] is not None:
        model = _memo["model"]
    else:
        model = parse(frame)
    new, changed = step(model, aid)
    out = render(new, frame)
    if changed:
        tick_hud(out)
    _memo["frame"], _memo["model"] = [list(r) for r in out], new
    return out
