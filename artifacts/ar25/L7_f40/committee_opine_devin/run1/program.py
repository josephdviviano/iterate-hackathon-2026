# Mechanics: two-axis mirror board of 21x21 3px blocks (ring + centre); H axis row / V axis column of wall 10,
# pieces 5, targets 11 (solid, on top of walls), background 9. A5 cycles selection H -> V -> pieces (min row, col) -> H;
# A1/A2 move H or piece up/down, A3/A4 move V or piece left/right; pieces blocked by board edge and other pieces, axes only by edge.
# Every piece block mirrors across V, H and both axes as visible 4 blocks (centre hole shows lower solids), clipped to the board, not drawn on pieces.
# Col 63 counts effective actions as 12s from the top (blocked moves are free). Unconfirmed: A5 from the last piece back to H; target blocking.
BG, WALL, PIECE, REFL, TARGET, SEL, HUD = 9, 10, 5, 4, 11, 0, 12
N = 21
_memo = {"frame": None, "model": None}


def _ring(f, bx, by):
    return f[3 * by][3 * bx]


def _centre(f, bx, by):
    return f[3 * by + 1][3 * bx + 1]


def _components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c = stack.pop()
            comp.add(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in cells:
                    cells.discard(n)
                    stack.append(n)
        comps.append(frozenset(comp))
    return comps


def _order(pieces):
    return sorted(pieces, key=lambda p: min((by, bx) for bx, by in p))


def parse(f):
    rows = [sum(_ring(f, bx, by) == WALL for bx in range(N)) for by in range(N)]
    cols = [sum(_ring(f, bx, by) == WALL for by in range(N)) for bx in range(N)]
    ah, av = rows.index(max(rows)), cols.index(max(cols))
    pieces = _order(_components((bx, by) for by in range(N) for bx in range(N) if _ring(f, bx, by) == PIECE))
    targets = {(bx, by) for by in range(N) for bx in range(N) if _centre(f, bx, by) == TARGET}
    covered = set().union(*pieces) if pieces else set()
    sel = None
    for bx in range(N):
        if bx != av and (bx, ah) not in covered and _ring(f, bx, ah) == WALL and _centre(f, bx, ah) == SEL:
            sel = "H"
    if sel is None:
        for by in range(N):
            if by != ah and (av, by) not in covered and _ring(f, av, by) == WALL and _centre(f, av, by) == SEL:
                sel = "V"
    if sel is None:
        for i, p in enumerate(pieces):
            if any(_centre(f, bx, by) == SEL for bx, by in p):
                sel = i
    return {"ah": ah, "av": av, "pieces": pieces, "targets": targets, "sel": sel}


def _inside(c):
    return 0 <= c[0] < N and 0 <= c[1] < N


def step(m, action):
    m = dict(m)
    pieces, sel = list(m["pieces"]), m["sel"]
    if action == 5:
        order = ["H", "V"] + list(range(len(pieces)))
        m["sel"] = order[(order.index(sel) + 1) % len(order)] if sel in order else "H"
        return m, True
    d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action)
    if d is None or sel is None:
        return m, False
    if sel == "H":
        if d[1] == 0 or not 0 <= m["ah"] + d[1] < N:
            return m, False
        m["ah"] += d[1]
        return m, True
    if sel == "V":
        if d[0] == 0 or not 0 <= m["av"] + d[0] < N:
            return m, False
        m["av"] += d[0]
        return m, True
    moved = frozenset((bx + d[0], by + d[1]) for bx, by in pieces[sel])
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
    if not all(_inside(c) for c in moved) or moved & others:
        return m, False
    pieces[sel] = moved
    m["pieces"] = pieces
    return m, True


def mirrors(m):
    ah, av = m["ah"], m["av"]
    covered = set().union(*m["pieces"]) if m["pieces"] else set()
    vis = set()
    for p in m["pieces"]:
        for bx, by in p:
            if bx != av:
                vis.add((2 * av - bx, by))
            if by != ah:
                vis.add((bx, 2 * ah - by))
            if bx != av and by != ah:
                vis.add((2 * av - bx, 2 * ah - by))
    return {c for c in vis if _inside(c) and c not in covered}


def render_board(m):
    vis = mirrors(m)
    sel = m["sel"]
    piece_of = {}
    for i, p in enumerate(m["pieces"]):
        for c in p:
            piece_of[c] = i
    out = [[BG] * 63 for _ in range(63)]
    for by in range(N):
        for bx in range(N):
            c = (bx, by)
            stack = []  # (ring colour or None for eraser, solid centre?, centre fill or None)
            if c in piece_of:
                stack.append((PIECE, False, SEL if sel == piece_of[c] else None))
            if c in vis:
                stack.append((REFL, False, REFL))
            if c in m["targets"]:
                stack.append((TARGET, True, TARGET))
            on_v, on_h = bx == m["av"], by == m["ah"]
            if on_v or on_h:
                lit = (on_v and sel == "V") or (on_h and sel == "H")
                stack.append((WALL, lit, SEL if lit else None))
            ring, centre = BG, BG
            if stack:
                ring = stack[0][0]
                solid = [s for s in stack if s[1]]
                if solid:
                    centre = solid[0][2]
                elif stack[0][2] is not None:
                    centre = stack[0][2]
            for dy in range(3):
                for dx in range(3):
                    out[3 * by + dy][3 * bx + dx] = centre if (dx, dy) == (1, 1) else ring
    return out


def transition_function(state, action, frame):
    aid = action.get("action_id") if isinstance(action, dict) else action
    if _memo["frame"] is not None and frame == _memo["frame"]:
        model = _memo["model"]
    else:
        model = parse(frame)
    new, effective = step(model, aid)
    out = [list(map(int, row)) for row in frame]
    board = render_board(new)
    for y in range(63):
        out[y][:63] = board[y]
    count = sum(1 for y in range(64) if frame[y][63] == HUD)
    if effective:
        count = min(64, count + 1)
    for y in range(63):
        out[y][63] = HUD if y < count else TARGET
    if count == 64:
        out[63][63] = HUD
    _memo["frame"], _memo["model"] = out, new
    return out
