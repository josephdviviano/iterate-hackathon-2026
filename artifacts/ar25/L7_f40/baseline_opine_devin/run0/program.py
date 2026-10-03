# Mechanics: 21x21 board of 3x3 blocks (bg 9). Two wall axes (colour 10): H band (block row ah), V strip (block col av).
# Pieces (ring 5) mirror across V, H and both into solid 4 blocks (on-axis sources, off-board and piece cells skipped).
# Stack piece > reflection > target (solid 11) > walls; selected entity shows 0 centres. A5 cycles H -> V -> pieces by
# min block (row, col). A1/A2 move H or piece up/down, A3/A4 move V or piece left/right; pieces blocked by edge/other
# pieces. HUD col 63: first non-12 cell -> 12 per effective action. Unconfirmed: piece on an axis row (never observed).
BG, WALL, TGT, PIECE, REFL, SEL, HUD_ON = 9, 10, 11, 5, 4, 0, 12
N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def block(frame, r, c):
    return [frame[3 * r + i][3 * c + j] for i in range(3) for j in range(3)]


def parse(frame):
    blocks = {(r, c): block(frame, r, c) for r in range(N) for c in range(N)}
    is_wall = {k for k, px in blocks.items() if px[0] == WALL}
    rows = [sum((r, c) in is_wall for c in range(N)) for r in range(N)]
    cols = [sum((r, c) in is_wall for r in range(N)) for c in range(N)]
    ah = max(range(N), key=lambda r: rows[r])
    av = max(range(N), key=lambda c: cols[c])
    targets = {k for k, px in blocks.items() if TGT in px}
    pcells = {k for k, px in blocks.items() if px[0] == PIECE}
    pieces = components(pcells)
    sel = None
    hdots = sum(1 for (r, c) in is_wall if r == ah and c != av and blocks[(r, c)][4] == SEL)
    vdots = sum(1 for (r, c) in is_wall if c == av and r != ah and blocks[(r, c)][4] == SEL)
    if hdots or vdots:
        sel = 'H' if hdots >= vdots else 'V'
    else:
        for i, p in enumerate(pieces):
            if any(blocks[k][4] == SEL for k in p):
                sel = i
    return {'ah': ah, 'av': av, 'targets': targets, 'pieces': pieces, 'sel': sel}


def components(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        out.append(frozenset(comp))
    return sorted(out, key=lambda p: min(p))


def on_board(k):
    return 0 <= k[0] < N and 0 <= k[1] < N


def step(m, action):
    m = dict(m, pieces=list(m['pieces']))
    aid = action.get('action_id') if isinstance(action, dict) else action
    sel = m['sel']
    if aid == 5:
        order = ['H', 'V'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return m, m['sel'] != sel
    if aid not in DIRS or sel is None:
        return m, False
    dr, dc = DIRS[aid]
    if sel == 'H':
        nr = m['ah'] + dr
        if dc or not 0 <= nr < N:
            return m, False
        m['ah'] = nr
        return m, True
    if sel == 'V':
        nc = m['av'] + dc
        if dr or not 0 <= nc < N:
            return m, False
        m['av'] = nc
        return m, True
    moved = frozenset((r + dr, c + dc) for r, c in m['pieces'][sel])
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != sel])
    if not all(on_board(k) for k in moved) or moved & others:
        return m, False
    m['pieces'][sel] = moved
    return m, True


def mirrors(m):
    """Every piece block mirrors across V, across H and across both (sources on that axis skipped)."""
    ah, av = m['ah'], m['av']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    vis = set()
    for p in m['pieces']:
        for r, c in p:
            if c != av:
                vis.add((r, 2 * av - c))
            if r != ah:
                vis.add((2 * ah - r, c))
                if c != av:
                    vis.add((2 * ah - r, 2 * av - c))
    return {k for k in vis if on_board(k) and k not in occ}


def layers(m, k, vis):
    """Top-down list of (ring colour or None, centre: ('solid'|'fill'|None, colour))."""
    out = []
    for i, p in enumerate(m['pieces']):
        if k in p:
            out.append((PIECE, ('fill', SEL) if m['sel'] == i else None))
    if k in vis:
        out.append((REFL, ('fill', REFL)))
    if k in m['targets']:
        out.append((TGT, ('solid', TGT)))
    if k[1] == m['av']:
        out.append((WALL, ('solid', SEL) if m['sel'] == 'V' or (m['sel'] == 'H' and k[0] == m['ah']) else None))
    if k[0] == m['ah']:
        out.append((WALL, ('solid', SEL) if m['sel'] == 'H' else None))
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    vis = mirrors(m)
    for r in range(N):
        for c in range(N):
            ls = layers(m, (r, c), vis)
            ring = ls[0][0] if ls else BG
            centre = next((v for _, cen in ls if cen and cen[0] == 'solid' for v in [cen[1]]), None)
            if centre is None:
                centre = next((cen[1] for _, cen in ls if cen), BG)
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = centre if (i, j) == (1, 1) else ring
    return out


def tick_hud(out):
    for y in range(64):
        if out[y][63] != HUD_ON:
            out[y][63] = HUD_ON
            return


def transition_function(state, action, frame):
    m = parse(frame)
    m2, changed = step(m, action)
    if not changed:
        return [list(row) for row in frame]
    out = render(m2, frame)
    tick_hud(out)
    return out
