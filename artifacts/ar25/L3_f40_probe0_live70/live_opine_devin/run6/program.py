# Mechanics: 21x21 board of 3x3 blocks; axis band (10) on one block row, targets (11, solid), reflections (4),
# pieces (5 rings, holes). A1/A2 move the selection a block row (axis or piece), A3/A4 move a piece a column
# (axis ignores them); pieces are blocked by the board edge and other pieces, never by the axis. A5 cycles
# axis -> pieces by size desc -> axis. Each piece block off the axis row reflects to row 2A-r (skip piece blocks).
# HUD col 63: n effective actions; n<=64 -> n cells 12 from top, else n-64 cells 5 then 12. Unconfirmed: n>=128.
N = 21
BG, AX, TG, RF, PC, SEL, HUDON, HUDOFF, SPENT = 9, 10, 11, 4, 5, 0, 12, 11, 5
_memo = {"frame": None, "model": None}


def blk(frame, r, c, dy=1, dx=1):
    return frame[3 * r + dy][3 * c + dx]


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in cells:
                    cells.discard(q)
                    comp.add(q)
                    st.append(q)
        out.append(comp)
    return out


def parse(frame):
    cnt = [sum(blk(frame, r, c, 0, 0) == AX for c in range(N)) for r in range(N)]
    axis = max(range(N), key=lambda r: cnt[r])
    axis_sel = any(blk(frame, axis, c, 0, 0) == AX and blk(frame, axis, c) == SEL for c in range(N))
    targets = {(r, c) for r in range(N) for c in range(N)
               if any(frame[3 * r + i][3 * c + j] == TG for i in range(3) for j in range(3))}
    pblocks = {(r, c) for r in range(N) for c in range(N) if blk(frame, r, c, 0, 0) == PC}
    seeds = set() if axis_sel else {b for b in pblocks if blk(frame, *b) == SEL}
    sel = set(seeds)
    st = list(seeds)
    while st:
        r, c = st.pop()
        for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if q in pblocks and q not in sel and blk(frame, *q) != BG:
                sel.add(q)
                st.append(q)
    pieces = []
    if sel:
        pieces += comps(sel)
    pieces += comps(pblocks - sel)
    seli = None
    if sel:
        seli = 0
        if len(pieces) > 1 and not pieces[0] & seeds:
            seli = None
    return {"axis": axis, "sel": "axis" if axis_sel else seli, "pieces": pieces, "targets": targets}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    a = action if isinstance(action, int) else action.get("action_id")
    axis, sel, pieces = m["axis"], m["sel"], [set(p) for p in m["pieces"]]
    if a == 5:
        seq = ["axis"] + order(pieces)
        k = seq.index(sel) if sel in seq else 0
        sel = seq[(k + 1) % len(seq)]
    elif a in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a]
        if sel == "axis":
            if dc == 0 and 0 <= axis + dr < N:
                axis += dr
        elif sel is not None:
            moved = {(r + dr, c + dc) for r, c in pieces[sel]}
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel])
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not moved & others:
                pieces[sel] = moved
    return {"axis": axis, "sel": sel, "pieces": pieces, "targets": m["targets"]}


def same(m1, m2):
    return (m1["axis"] == m2["axis"] and m1["sel"] == m2["sel"]
            and [sorted(p) for p in m1["pieces"]] == [sorted(p) for p in m2["pieces"]])


def reflections(m):
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    A, out = m["axis"], set()
    for r, c in occ:
        if r != A and 0 <= 2 * A - r < N and (2 * A - r, c) not in occ:
            out.add((2 * A - r, c))
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
            stack = []  # (ring colour, solid centre or None, fill or None), top-down
            if (r, c) in owner:
                stack.append((PC, None, SEL if m["sel"] == owner[(r, c)] else None))
            if (r, c) in refl:
                stack.append((RF, None, RF))
            if (r, c) in m["targets"]:
                stack.append((TG, TG, None))
            if r == m["axis"]:
                stack.append((AX, SEL if m["sel"] == "axis" else None, None))
            ring = stack[0][0] if stack else BG
            centre = next((s for _, s, _ in stack if s is not None), None)
            if centre is None:
                centre = next((f for _, _, f in stack if f is not None), BG)
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = centre if (i, j) == (1, 1) else ring
    return out


def hud_count(frame):
    col = [frame[y][63] for y in range(64)]
    spent = 0
    while spent < 64 and col[spent] == SPENT:
        spent += 1
    return 64 + spent if spent else sum(v == HUDON for v in col)


def draw_hud(out, n):
    for y in range(64):
        if n <= 64:
            out[y][63] = HUDON if y < n else HUDOFF
        else:
            out[y][63] = SPENT if y < n - 64 else HUDON


def transition_function(state, action, frame):
    if _memo["frame"] is not None and frame == _memo["frame"]:
        m = _memo["model"]
    else:
        m = parse(frame)
    m2 = step(m, action)
    out = render(m2, frame)
    n = hud_count(frame)
    if not same(m, m2):
        n += 1
    draw_hud(out, n)
    _memo["frame"] = [list(r) for r in out]
    _memo["model"] = m2
    return out
