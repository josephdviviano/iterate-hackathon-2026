# Mechanics: frame-space mirror-axis game on a 21x21 grid of 3x3 blocks (board 63x63; col 63 = HUD, row 63 static).
# Colours: bg 9, axis band 10 (layer 1), targets 11 solid (2), reflections 4 holed (3), pieces 5 holed (4).
# A1/A2 move the selection one block row up/down (axis passes under pieces; pieces blocked by board/other pieces);
# A3/A4 move a selected piece left/right (no-op for axis); A5 cycles axis -> pieces by size desc -> axis.
# Reflections: block row r -> 2A-r, skipping on-axis sources, off-board and piece-covered dests. HUD: one more colour-12 cell per state-changing action. 'player gone' = touching pieces merging (render artefact), split via 0-centres.
N = 21
_memo = {"frame": None, "model": None}


def _ring(frame, bx, by):
    return [frame[3 * by + j][3 * bx + i] for j in range(3) for i in range(3) if (i, j) != (1, 1)]


def _centre(frame, bx, by):
    return frame[3 * by + 1][3 * bx + 1]


def _comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def parse(frame):
    axis, axis_sel, targets, pblocks, centres = None, False, set(), set(), {}
    for by in range(N):
        for bx in range(N):
            ring, c = set(_ring(frame, bx, by)), _centre(frame, bx, by)
            if ring == {10}:
                axis = by
                axis_sel = axis_sel or c == 0
            elif ring == {11}:
                targets.add((bx, by))
            elif ring == {5}:
                pblocks.add((bx, by))
                centres[(bx, by)] = c
            if ring in ({5}, {4}) and c == 11:
                targets.add((bx, by))
    pieces, sel = [], None
    for comp in _comps(pblocks):
        zeros = {b for b in comp if centres[b] == 0 and not (axis_sel and b[1] == axis)}
        if not zeros:
            pieces.append(comp)
            continue
        grown, st = set(zeros), list(zeros)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in comp and n not in grown and centres[n] != 9:
                    grown.add(n)
                    st.append(n)
        sel = len(pieces)
        pieces.append(grown)
        pieces.extend(_comps(comp - grown))
    return {"axis": axis, "sel": "axis" if sel is None else sel, "pieces": pieces, "targets": targets}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    m = {"axis": m["axis"], "sel": m["sel"], "pieces": [set(p) for p in m["pieces"]], "targets": m["targets"]}
    sel = m["sel"]
    if action == 5:
        seq = ["axis"] + order(m["pieces"])
        m["sel"] = seq[(seq.index(sel) + 1) % len(seq)]
        return m, m["sel"] != sel
    d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action)
    if d is None:
        return m, False
    if sel == "axis":
        if d[0] or not 0 <= m["axis"] + d[1] < N:
            return m, False
        m["axis"] += d[1]
        return m, True
    others = set().union(*[p for i, p in enumerate(m["pieces"]) if i != sel])
    moved = {(x + d[0], y + d[1]) for x, y in m["pieces"][sel]}
    if any(not (0 <= x < N and 0 <= y < N) for x, y in moved) or moved & others:
        return m, False
    m["pieces"][sel] = moved
    return m, True


def render(m, frame):
    out = [row[:] for row in frame]
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    A = m["axis"]
    refl = set()
    for p in m["pieces"]:
        for x, y in p:
            if y != A and 0 <= 2 * A - y < N and (x, 2 * A - y) not in occupied:
                refl.add((x, 2 * A - y))
    for by in range(N):
        for bx in range(N):
            b = (bx, by)
            layers = []  # top-down: (ring colour, centre solid or None, centre fill or None)
            for i, p in enumerate(m["pieces"]):
                if b in p:
                    layers.append((5, None, 0 if m["sel"] == i else None))
            if b in refl:
                layers.append((4, None, 4))
            if b in m["targets"]:
                layers.append((11, 11, None))
            if by == A:
                layers.append((10, 0 if m["sel"] == "axis" else None, None))
            ring = layers[0][0] if layers else 9
            centre, fill = None, None
            for _, solid, f in layers:
                if solid is not None:
                    centre = solid
                    break
                if fill is None and f is not None:
                    fill = f
            if centre is None:
                centre = fill if fill is not None else 9
            for j in range(3):
                for i in range(3):
                    out[3 * by + j][3 * bx + i] = centre if (i, j) == (1, 1) else ring
    return out


def transition_function(state, action, frame):
    if _memo["frame"] == frame and _memo["model"] is not None:
        m = _memo["model"]
    else:
        m = parse(frame)
    aid = action["action_id"] if isinstance(action, dict) else action
    m2, changed = step(m, aid)
    out = render(m2, frame)
    k = sum(1 for y in range(63) if frame[y][63] == 12) + (1 if changed else 0)
    for y in range(63):
        out[y][63] = 12 if y < k else 11
    _memo["frame"], _memo["model"] = out, m2
    return out
