# Mechanics: 21x21 grid of 3x3 blocks (board 63x63); horizontal axis band (colour 10), targets (11),
# pieces (5, holed blocks) and their mirror images (4) across the axis (dest row 2A-r, skipped
# off-board / on piece blocks / for blocks on the axis row). A1/A2 move the selected thing up/down 3,
# A3/A4 move a selected piece left/right (blocked by board edge / other pieces), A5 cycles axis ->
# pieces (size desc) -> axis. Holes: selected 0, reflection 4, else transparent. Col 63 counts effective
# actions in colour 12. Hypotheses unconfirmed: axis may pass under pieces; win/budget-full behaviour.
N, B = 63, 21
BG, WALL, TGT, REFL, PIECE, SEL, CNT, CBG = 9, 10, 11, 4, 5, 0, 12, 11
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memo = {"frame": None, "model": None}


def cells(br, bc):
    return [(3 * br + i, 3 * bc + j) for i in range(3) for j in range(3)]


def border(frame, br, bc):
    return [frame[y][x] for (y, x) in cells(br, bc) if (y, x) != (3 * br + 1, 3 * bc + 1)]


def centre(frame, br, bc):
    return frame[3 * br + 1][3 * bc + 1]


def components(blocks):
    blocks, comps = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + dr, b[1] + dc)
                if n in blocks:
                    blocks.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def split_merged(comp, frame, axis_sel):
    """A component holding both a selected (0 hole) and unselected (bg hole) piece is split."""
    if axis_sel:
        return [comp]
    sel = {b for b in comp if centre(frame, *b) == SEL}
    uns = {b for b in comp if centre(frame, *b) == BG}
    if not sel or not uns:
        return [comp]
    rest = comp - sel - uns
    groups = [sel, uns]
    changed = True
    while rest and changed:
        changed = False
        for b in list(rest):
            for g in groups:
                if any(abs(b[0] - o[0]) + abs(b[1] - o[1]) == 1 for o in g):
                    g.add(b)
                    rest.discard(b)
                    changed = True
                    break
    if rest:
        groups[0] |= rest
    return groups


def parse(frame):
    axis = None
    for br in range(B):
        if any(all(v == WALL for v in border(frame, br, bc)) for bc in range(B)):
            axis = br
            break
    axis_sel = axis is not None and any(
        all(v == WALL for v in border(frame, axis, bc)) and centre(frame, axis, bc) == SEL
        for bc in range(B))
    piece_blocks = [(br, bc) for br in range(B) for bc in range(B)
                    if all(v == PIECE for v in border(frame, br, bc))]
    targets = {(br, bc) for br in range(B) for bc in range(B)
               if any(frame[y][x] == TGT for (y, x) in cells(br, bc))}
    pieces = []
    for comp in components(piece_blocks):
        pieces.extend(split_merged(comp, frame, axis_sel))
    pieces.sort(key=lambda p: (-len(p), min(p)))
    sel = "axis"
    if not axis_sel:
        for i, p in enumerate(pieces):
            if any(centre(frame, *b) == SEL for b in p):
                sel = i
                break
    return {"axis": axis, "sel": sel, "pieces": [frozenset(p) for p in pieces],
            "targets": frozenset(targets)}


def step(m, action):
    m = dict(m, pieces=list(m["pieces"]))
    if action == 5:
        order = sorted(range(len(m["pieces"])), key=lambda i: (-len(m["pieces"][i]), min(m["pieces"][i])))
        seq = ["axis"] + order
        m["sel"] = seq[(seq.index(m["sel"]) + 1) % len(seq)] if m["sel"] in seq else "axis"
        return m
    if action not in MOVES:
        return m
    dr, dc = MOVES[action]
    if m["sel"] == "axis":
        if dc == 0 and m["axis"] is not None and 0 <= m["axis"] + dr < B:
            m["axis"] += dr
        return m
    i = m["sel"]
    moved = frozenset((r + dr, c + dc) for r, c in m["pieces"][i])
    others = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i])
    if all(0 <= r < B and 0 <= c < B for r, c in moved) and not (moved & others):
        m["pieces"][i] = moved
    return m


def reflections(m):
    a = m["axis"]
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    if a is None:
        return out
    for p in m["pieces"]:
        for r, c in p:
            d = 2 * a - r
            if r != a and 0 <= d < B and (d, c) not in occupied:
                out.add((d, c))
    return out


def render(m, frame):
    out = [row[:] for row in frame]
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m["pieces"]):
        for b in p:
            owner[b] = i
    for br in range(B):
        for bc in range(B):
            b = (br, bc)
            stack = []  # top-down: (border colour, hole fill or None for solid centre)
            if b in owner:
                stack.append((PIECE, SEL if m["sel"] == owner[b] else -1))
            if b in refl:
                stack.append((REFL, REFL))
            if b in m["targets"]:
                stack.append((TGT, None))
            if br == m["axis"]:
                stack.append((WALL, None if m["sel"] == "axis" else -1))
            bcol = stack[0][0] if stack else BG
            ccol = BG
            fill = None
            for col, hole in stack:
                if hole is None:
                    ccol = SEL if col == WALL else col
                    break
                if hole >= 0 and fill is None:
                    fill = hole
            else:
                ccol = fill if fill is not None else BG
            for (y, x) in cells(br, bc):
                out[y][x] = ccol if (y, x) == (3 * br + 1, 3 * bc + 1) else bcol
    return out


def same_model(a, b):
    return (a["axis"] == b["axis"] and a["sel"] == b["sel"] and a["pieces"] == b["pieces"])


def transition_function(state, action, frame=None):
    aid = action["action_id"] if isinstance(action, dict) else action
    if _memo["frame"] is not None and _memo["frame"] == frame:
        m = _memo["model"]
    else:
        m = parse(frame)
    new = step(m, aid)
    out = render(new, frame)
    count = sum(1 for y in range(len(frame)) if frame[y][N] == CNT)
    if not same_model(m, new):
        count += 1
    for y in range(len(frame)):
        out[y][N] = CNT if y < count else CBG
    _memo["frame"], _memo["model"] = out, new
    return out
