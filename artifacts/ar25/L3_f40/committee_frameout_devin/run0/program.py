# Mechanics: 21x21 board of 3x3 blocks; layers wall(axis,10) < target(11) < reflection(4) < piece(5).
# A1/A2 move the selection up/down 3 rows, A3/A4 left/right (axis ignores A3/A4); A5 cycles axis -> pieces
# (size desc) -> axis. Pieces are blocked by the board edge and other pieces; the axis is not blocked.
# Every piece block off the axis row reflects to row 2a-b unless off-board or covered by a piece.
# Selected sprite shows 0 in its holes; HUD column 63 counts effective actions (colour 12). Unconfirmed: budget overflow.
N = 21
BG, WALL, TGT, REFL, PIECE, SEL, HUDON = 9, 10, 11, 4, 5, 0, 12
_memo = {"frame": None, "model": None}


def _block(frame, bx, by):
    return [frame[3 * by + j][3 * bx + i] for j in range(3) for i in range(3)]


def _ring(cells):
    return cells[:4] + cells[5:]


def parse(frame):
    axis = None
    for by in range(N):
        if any(WALL in _ring(_block(frame, bx, by)) for bx in range(N)):
            axis = by
            break
    axis_sel = False
    piece_blocks, sel_blocks, targets = set(), set(), set()
    for by in range(N):
        for bx in range(N):
            c = _block(frame, bx, by)
            ring = _ring(c)
            if all(v == PIECE for v in ring):
                piece_blocks.add((bx, by))
                if c[4] == SEL and not (by == axis and _wall_dot_selected(frame, axis)):
                    sel_blocks.add((bx, by))
                if c[4] == TGT:
                    targets.add((bx, by))
            elif all(v == WALL for v in ring) and c[4] == SEL:
                axis_sel = True
            elif TGT in c:
                targets.add((bx, by))
    comps = _components(piece_blocks)
    pieces = []
    for comp in comps:
        s = _grow(comp & sel_blocks, comp, frame)
        if s and s != comp:
            pieces.append(frozenset(s))
            pieces.extend(frozenset(c) for c in _components(comp - s))
        else:
            pieces.append(frozenset(comp))
    pieces = _order(pieces)
    sel = -1 if axis_sel or not sel_blocks else next(
        i for i, p in enumerate(pieces) if p & sel_blocks)
    return {"axis": axis, "sel": sel, "pieces": pieces, "targets": frozenset(targets)}


def _grow(seed, comp, frame):
    s, stack = set(seed), list(seed)
    while stack:
        x, y = stack.pop()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n in comp and n not in s and _block(frame, *n)[4] != BG:
                s.add(n)
                stack.append(n)
    return s


def _wall_dot_selected(frame, axis):
    for bx in range(N):
        c = _block(frame, bx, axis)
        if all(v == WALL for v in _ring(c)):
            return c[4] == SEL
    return False


def _order(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min((y, x) for x, y in p)))


def _components(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c = stack.pop()
            comp.add(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def step(m, action):
    m = dict(m)
    if action == 5:
        m["sel"] = m["sel"] + 1 if m["sel"] + 1 < len(m["pieces"]) else -1
        return m
    if action not in MOVES:
        return m
    dx, dy = MOVES[action]
    if m["sel"] < 0:
        if dx == 0 and m["axis"] is not None and 0 <= m["axis"] + dy < N:
            m["axis"] += dy
        return m
    p = m["pieces"][m["sel"]]
    moved = frozenset((x + dx, y + dy) for x, y in p)
    others = set().union(*[q for i, q in enumerate(m["pieces"]) if i != m["sel"]])
    if all(0 <= x < N and 0 <= y < N for x, y in moved) and not (moved & others):
        m["pieces"] = list(m["pieces"])
        m["pieces"][m["sel"]] = moved
    return m


def reflections(m):
    a = m["axis"]
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    if a is None:
        return out
    for x, y in occupied:
        if y == a:
            continue
        d = (x, 2 * a - y)
        if 0 <= d[1] < N and d not in occupied:
            out.add(d)
    return out


def render(m, frame):
    out = [row[:] for row in frame]
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m["pieces"]):
        for b in p:
            owner[b] = i
    for by in range(N):
        for bx in range(N):
            b = (bx, by)
            layers = []
            if b in owner:
                layers.append(("piece", owner[b] == m["sel"]))
            if b in refl:
                layers.append(("refl", False))
            if b in m["targets"]:
                layers.append(("tgt", False))
            if by == m["axis"]:
                layers.append(("wall", m["sel"] == -1))
            ring = {"piece": PIECE, "refl": REFL, "tgt": TGT, "wall": WALL}[layers[0][0]] if layers else BG
            centre = BG
            if layers and layers[0] == ("piece", True):
                centre = SEL
            elif any(k == "refl" for k, _ in layers):
                centre = REFL
            for k, s in layers:
                if k == "tgt" or (k == "wall" and s):
                    centre = TGT if k == "tgt" else SEL
                    break
            for j in range(3):
                for i in range(3):
                    out[3 * by + j][3 * bx + i] = centre if (i, j) == (1, 1) else ring
    return out


def transition_function(state, action, frame):
    if isinstance(action, dict):
        action = action.get("action_id")
    if _memo["frame"] is not None and frame == _memo["frame"]:
        model = _memo["model"]
    else:
        model = parse(frame)
    new = step(model, action)
    out = render(new, frame)
    if out != frame:
        n = sum(1 for y in range(64) if frame[y][63] == HUDON)
        if n < 64:
            out[n][63] = HUDON
    _memo["frame"], _memo["model"] = out, new
    return out
