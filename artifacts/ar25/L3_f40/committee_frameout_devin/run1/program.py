# Mechanics: 21x21 board of 3x3 blocks (bg 9); one horizontal wall axis (ring 10) on block row A; pieces (ring 5) hole-centred;
# targets solid 11. Selection = axis (wall centres 0) or a piece (its centres 0); A5 cycles axis -> pieces by size desc -> axis.
# A1/A2 move the selected axis/piece up/down, A3/A4 move a piece left/right (axis ignores them); bounds + other pieces block pieces.
# Every piece block (r,c) with r != A reflects to (2A-r,c) as a ring-4 sprite with fill 4, skipped off-board / on piece blocks.
# HUD: column 63 gains one 12 cell per state-changing action. Unconfirmed: blocked piece moves / no-ops tick? win/reset rules.
BG, WALL, PIECE, TARGET, REFL, DOT, HUDON = 9, 10, 5, 11, 4, 0, 12
N = 21
_memo = {"frame": None, "state": None}


def cell(f, bx, by, dx, dy):
    return f[3 * by + dy][3 * bx + dx]


def parse(frame):
    axis = None
    for y in range(3 * N):
        if WALL in frame[y][:3 * N]:
            axis = y // 3
            break
    axis_sel = axis is not None and any(
        cell(frame, bx, axis, 0, 0) == WALL and cell(frame, bx, axis, 1, 1) == DOT for bx in range(N))
    targets = {(bx, by) for by in range(N) for bx in range(N)
               if cell(frame, bx, by, 1, 1) == TARGET or cell(frame, bx, by, 0, 0) == TARGET}
    pblocks = {(bx, by) for by in range(N) for bx in range(N) if cell(frame, bx, by, 0, 0) == PIECE}
    pieces, sel = [], None
    for comp in components(pblocks):
        zero = {b for b in comp if cell(frame, b[0], b[1], 1, 1) == DOT}
        if axis_sel or not zero:
            pieces.append(comp)
            continue
        amb = {b for b in comp if cell(frame, b[0], b[1], 1, 1) not in (BG, DOT)}
        chosen = set()
        for c in components(zero | amb):
            if c & zero:
                chosen |= c
        pieces.append(frozenset(chosen))
        sel = len(pieces) - 1
        for rest in components(set(comp) - chosen):
            pieces.append(rest)
    return {"axis": axis, "sel": "axis" if axis_sel else sel, "pieces": pieces, "targets": targets}


def components(blocks):
    blocks, out = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for nb in ((b[0] + 1, b[1]), (b[0] - 1, b[1]), (b[0], b[1] + 1), (b[0], b[1] - 1)):
                if nb in blocks:
                    blocks.remove(nb)
                    stack.append(nb)
        out.append(frozenset(comp))
    out.sort(key=lambda c: min((b[1], b[0]) for b in c))
    return out


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min((b[1], b[0]) for b in pieces[i])))


def step(st, action):
    st = dict(st, pieces=list(st["pieces"]))
    sel = st["sel"]
    if action == 5:
        order = cycle_order(st["pieces"])
        if sel == "axis" or sel is None:
            st["sel"] = order[0] if order else "axis"
        else:
            k = order.index(sel)
            st["sel"] = order[k + 1] if k + 1 < len(order) else "axis"
        return st
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    if action not in moves or sel is None:
        return st
    dx, dy = moves[action]
    if sel == "axis":
        if dx == 0 and st["axis"] is not None and 0 <= st["axis"] + dy < N:
            st["axis"] += dy
        return st
    others = set().union(*[p for i, p in enumerate(st["pieces"]) if i != sel])
    moved = frozenset((x + dx, y + dy) for x, y in st["pieces"][sel])
    if all(0 <= x < N and 0 <= y < N for x, y in moved) and not (moved & others):
        st["pieces"][sel] = moved
    return st


def reflections(st):
    a = st["axis"]
    occupied = set().union(*st["pieces"]) if st["pieces"] else set()
    out = set()
    if a is None:
        return out
    for x, y in occupied:
        if y != a:
            ry = 2 * a - y
            if 0 <= ry < N and (x, ry) not in occupied:
                out.add((x, ry))
    return out


def render(st, frame):
    out = [row[:] for row in frame]
    refl = reflections(st)
    owner = {}
    for i, p in enumerate(st["pieces"]):
        for b in p:
            owner[b] = i
    for by in range(N):
        for bx in range(N):
            stack = []  # (ring colour, solid centre or None, hole fill or None), top-down
            b = (bx, by)
            if b in owner:
                stack.append((PIECE, None, DOT if st["sel"] == owner[b] else None))
            if b in refl:
                stack.append((REFL, None, REFL))
            if b in st["targets"]:
                stack.append((TARGET, TARGET, None))
            if by == st["axis"]:
                stack.append((WALL, DOT if st["sel"] == "axis" else None, None))
            ring = stack[0][0] if stack else BG
            centre = next((s[1] for s in stack if s[1] is not None), None)
            if centre is None:
                centre = next((s[2] for s in stack if s[2] is not None), BG)
            for dy in range(3):
                for dx in range(3):
                    out[3 * by + dy][3 * bx + dx] = centre if (dx, dy) == (1, 1) else ring
    return out


def same(a, b):
    return a["axis"] == b["axis"] and a["sel"] == b["sel"] and \
        sorted(map(sorted, a["pieces"])) == sorted(map(sorted, b["pieces"]))


def transition_function(state, action, frame=None):
    if isinstance(action, dict):
        action = action.get("action_id")
    if _memo["frame"] is not None and frame == _memo["frame"]:
        st = _memo["state"]
    else:
        st = parse(frame)
    new = step(st, action)
    out = render(new, frame)
    if not same(st, new):
        for y in range(3 * N):
            if out[y][63] != HUDON:
                out[y][63] = HUDON
                break
    _memo["frame"], _memo["state"] = out, new
    return out
