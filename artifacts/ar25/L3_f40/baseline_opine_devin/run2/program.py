# Mechanics: board = 21x21 grid of 3x3 blocks (bg 9); axis = full block row of colour 10, targets solid 11,
# pieces ring 5 with hole, reflections ring 4. A1/A2 move the selection a block row up/down, A3/A4 move a selected
# piece a block column (axis: no-op); pieces are blocked by the board edge and other pieces, the axis only by the edge.
# A5 cycles axis -> pieces by size desc -> axis. Reflection of piece block row r is row 2A-r (skip axis row, off-board,
# piece-occupied). Holes: selected piece 0 / reflection 4 fills, target shows through; HUD col 63 ticks 11->12 per effective action.
N, BG, AXIS, TARGET, REFL, PIECE, SEL = 21, 9, 10, 11, 4, 5, 0

def blk(frame, r, c):
    return [frame[3 * r + i][3 * c + j] for i in range(3) for j in range(3)]

def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]; comp = set(st)
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.discard(n); comp.add(n); st.append(n)
        out.append(comp)
    return out

def parse(frame):
    axis_row = max(range(N), key=lambda r: sum(frame[3 * r][3 * c] == AXIS for c in range(N)))
    targets, piece_cells, centre = set(), set(), {}
    for r in range(N):
        for c in range(N):
            b = blk(frame, r, c)
            ring = b[:4] + b[5:]
            if TARGET in b:
                targets.add((r, c))
            if PIECE in ring:
                piece_cells.add((r, c)); centre[(r, c)] = b[4]
    axis_sel = any(frame[3 * axis_row + 1][3 * c + 1] == SEL for c in range(N))
    pieces, sel = [], None
    for comp in comps(piece_cells):
        dots = {p for p in comp if centre[p] == SEL}
        if not dots or axis_sel:
            pieces.append(comp); continue
        grown, st = set(dots), list(dots)
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in comp and n not in grown and centre[n] != BG:
                    grown.add(n); st.append(n)
        sel = len(pieces); pieces.append(grown)
        pieces.extend(comps(comp - grown))
    return {"A": axis_row, "axis_sel": axis_sel, "sel": sel, "pieces": pieces, "targets": targets}

def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))

def step(m, action):
    a = action if isinstance(action, int) else action.get("action_id")
    pieces = [set(p) for p in m["pieces"]]
    A, axis_sel, sel = m["A"], m["axis_sel"], m["sel"]
    if a == 5:
        order = cycle_order(pieces)
        if axis_sel or (sel is None and not axis_sel):
            if order:
                axis_sel, sel = False, order[0]
            else:
                axis_sel, sel = True, None
        else:
            k = order.index(sel)
            if k + 1 < len(order):
                sel = order[k + 1]
            else:
                axis_sel, sel = True, None
    elif a in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a]
        if axis_sel:
            if dc == 0 and 0 <= A + dr < N:
                A += dr
        elif sel is not None:
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
            moved = {(r + dr, c + dc) for r, c in pieces[sel]}
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
                pieces[sel] = moved
    return {"A": A, "axis_sel": axis_sel, "sel": sel, "pieces": pieces, "targets": m["targets"]}

def same(m1, m2):
    key = lambda m: (m["A"], m["axis_sel"], sorted(sorted(p) for p in m["pieces"]),
                     sorted(m["pieces"][m["sel"]]) if m["sel"] is not None else None)
    return key(m1) == key(m2)

def reflections(m):
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    A, out = m["A"], set()
    for r, c in occ:
        if r == A:
            continue
        d = 2 * A - r
        if 0 <= d < N and (d, c) not in occ:
            out.add((d, c))
    return out

def render(m, frame):
    out = [list(row) for row in frame]
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m["pieces"]):
        for b in p:
            owner[b] = i
    for r in range(N):
        for c in range(N):
            # layer stack top-down: (ring colour, centre solid colour or None, centre fill or None)
            stack = []
            if (r, c) in owner:
                stack.append((PIECE, None, SEL if owner[(r, c)] == m["sel"] else None))
            if (r, c) in refl:
                stack.append((REFL, None, REFL))
            if (r, c) in m["targets"]:
                stack.append((TARGET, TARGET, None))
            if r == m["A"]:
                stack.append((AXIS, SEL if m["axis_sel"] else None, None))
            ring = stack[0][0] if stack else BG
            mid = next((s for _, s, _ in stack if s is not None), None)
            if mid is None:
                mid = next((f for _, _, f in stack if f is not None), BG)
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = mid if (i, j) == (1, 1) else ring
    return out

def hud_tick(out):
    for y in range(63):
        if out[y][63] != 12:
            out[y][63] = 12
            return

def transition_function(state, action, frame):
    m = parse(frame)
    m2 = step(m, action)
    out = render(m2, frame)
    if not same(m, m2):
        hud_tick(out)
    return [[int(v) for v in row] for row in out]
