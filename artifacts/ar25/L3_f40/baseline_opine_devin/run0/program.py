# Mechanics: 63x63 board = 21x21 grid of 3x3 blocks (bg 9); one horizontal axis row (ring 10), solid targets (11),
# holed pieces (ring 5), reflections (ring 4, centre 4) of every off-axis piece block to row 2A-r (skipped off-board/onto pieces).
# Selection shown by 0 centres (axis dots or selected piece holes); A1/A2 move selection one block row, A3/A4 move a selected
# piece one block column (blocked by board edge / other pieces), A5 cycles axis -> pieces by size desc -> axis.
# HUD col 63: one 11->12 per effective action. Hypothesis unconfirmed: axis is never blocked by pieces; A6/A7 are no-ops.
N = 21
BG, AXIS, TGT, REFL, PIECE, SEL, HUD_ON = 9, 10, 11, 4, 5, 0, 12


def corner(frame, bx, by):
    return frame[3 * by][3 * bx]


def centre(frame, bx, by):
    return frame[3 * by + 1][3 * bx + 1]


def neighbours(c):
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def components(cells):
    cells, out = set(cells), []
    while cells:
        seed = cells.pop()
        comp, stack = {seed}, [seed]
        while stack:
            for n in neighbours(stack.pop()):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(frame):
    blocks = [(x, y) for y in range(N) for x in range(N)]
    rows = [sum(1 for x in range(N) if corner(frame, x, y) == AXIS) for y in range(N)]
    axis = max(range(N), key=lambda y: rows[y])
    targets = {b for b in blocks if corner(frame, *b) == TGT or centre(frame, *b) == TGT}
    piece_cells = {b for b in blocks if corner(frame, *b) == PIECE}
    axis_sel = any(corner(frame, x, axis) == AXIS and centre(frame, x, axis) == SEL for x in range(N))
    sel_set = set()
    if not axis_sel:
        seeds = [b for b in piece_cells if centre(frame, *b) == SEL]
        sel_set, stack = set(seeds), list(seeds)
        while stack:
            for n in neighbours(stack.pop()):
                if n in piece_cells and n not in sel_set and centre(frame, *n) != BG:
                    sel_set.add(n)
                    stack.append(n)
    pieces = [sel_set] if sel_set else []
    pieces += components(piece_cells - sel_set)
    sel = 0 if sel_set else None
    return {"axis": axis, "axis_sel": axis_sel, "targets": targets, "pieces": pieces, "sel": sel}


def on_board(c):
    return 0 <= c[0] < N and 0 <= c[1] < N


def move_piece(m, dx, dy):
    i = m["sel"]
    others = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i]) if len(m["pieces"]) > 1 else set()
    dest = {(x + dx, y + dy) for x, y in m["pieces"][i]}
    if all(on_board(c) for c in dest) and not (dest & others):
        m["pieces"][i] = dest
        return True
    return False


def cycle_selection(m):
    order = sorted(range(len(m["pieces"])), key=lambda j: -len(m["pieces"][j]))
    if m["axis_sel"]:
        if not order:
            return False
        m["axis_sel"], m["sel"] = False, order[0]
    else:
        k = order.index(m["sel"])
        if k + 1 < len(order):
            m["sel"] = order[k + 1]
        else:
            m["axis_sel"], m["sel"] = True, None
    return True


def step(m, action):
    if isinstance(action, dict):
        return False
    if action in (1, 2):
        d = -1 if action == 1 else 1
        if m["axis_sel"]:
            if 0 <= m["axis"] + d < N:
                m["axis"] += d
                return True
            return False
        return m["sel"] is not None and move_piece(m, 0, d)
    if action in (3, 4):
        d = -1 if action == 3 else 1
        return (not m["axis_sel"]) and m["sel"] is not None and move_piece(m, d, 0)
    if action == 5:
        return cycle_selection(m)
    return False


def reflections(m):
    a = m["axis"]
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    for x, y in occupied:
        if y == a:
            continue
        d = (x, 2 * a - y)
        if on_board(d) and d not in occupied:
            out.add(d)
    return out


def render(m, frame):
    out = [list(r) for r in frame]
    refl = reflections(m)
    sel_cells = m["pieces"][m["sel"]] if m["sel"] is not None else set()
    all_pieces = set().union(*m["pieces"]) if m["pieces"] else set()
    for by in range(N):
        for bx in range(N):
            b = (bx, by)
            layers = []  # (ring, fill) top-down; fill None = transparent hole
            if b in all_pieces:
                layers.append((PIECE, SEL if b in sel_cells else None))
            if b in refl:
                layers.append((REFL, REFL))
            if b in m["targets"]:
                layers.append((TGT, TGT))
            if by == m["axis"]:
                layers.append((AXIS, SEL if m["axis_sel"] else None))
            ring = layers[0][0] if layers else BG
            if b in m["targets"]:
                mid = TGT
            else:
                fills = [f for _, f in layers if f is not None]
                mid = fills[0] if fills else BG
            for dy in range(3):
                for dx in range(3):
                    out[3 * by + dy][3 * bx + dx] = mid if (dx, dy) == (1, 1) else ring
    return out


def render_hud(out, frame, changed):
    ticks = sum(1 for y in range(63) if frame[y][63] == HUD_ON)
    if changed:
        ticks += 1
    for y in range(63):
        out[y][63] = HUD_ON if y < ticks else TGT


def transition_function(state, action, frame):
    m = parse(frame)
    changed = step(m, action)
    out = render(m, frame)
    render_hud(out, frame, changed)
    return [[int(v) for v in row] for row in out]
