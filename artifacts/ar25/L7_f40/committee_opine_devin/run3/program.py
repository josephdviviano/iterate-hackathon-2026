# Mechanics: 21x21 board of 3x3 blocks; row axis H (A1/A2) and column axis V (A3/A4) colour 10, selected = 0 dots;
# pieces colour 5 (selected = 0 centre) move one block with A1-A4, blocked by edge / other pieces; A5 cycles
# H -> V -> pieces by (min row, min col) -> H. Each piece block mirrors across V, H and both into solid 4 blocks
# (skip on-axis sources, piece/off-board dests); targets 11 on top, centre-only when covered; HUD col 63 +1 12 per change.
# Unconfirmed: whether a piece touching an axis hides its mirror (earlier variants did; never exercised here).
N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def px(f, r, c, dy=1, dx=1):
    return f[3 * r + dy][3 * c + dx]


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if q in cells:
                    cells.discard(q)
                    comp.add(q)
                    st.append(q)
        out.append(comp)
    return out


def parse(f):
    corner = {(r, c): px(f, r, c, 0, 0) for r in range(N) for c in range(N)}
    ah = max(range(N), key=lambda r: sum(corner[(r, c)] == 10 for c in range(N)))
    av = max(range(N), key=lambda c: sum(corner[(r, c)] == 10 for r in range(N)))
    targets = {(r, c) for r in range(N) for c in range(N)
               if any(f[3 * r + dy][3 * c + dx] == 11 for dy in range(3) for dx in range(3))}
    hsel = any(corner[(ah, c)] == 10 and px(f, ah, c) == 0 for c in range(N) if c != av)
    vsel = any(corner[(r, av)] == 10 and px(f, r, av) == 0 for r in range(N) if r != ah)
    pblocks = {k for k, v in corner.items() if v == 5}
    pieces, sel = [], None
    if hsel:
        sel = 'H'
    elif vsel:
        sel = 'V'
    for comp in comps(pblocks):
        dots = {b for b in comp if px(f, *b) == 0}
        if sel is None and dots:
            grown, st = set(dots), list(dots)
            while st:
                r, c = st.pop()
                for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if q in comp and q not in grown and px(f, *q) != 9:
                        grown.add(q)
                        st.append(q)
            pieces.append(grown)
            sel = len(pieces) - 1
            for rest in comps(comp - grown):
                pieces.append(rest)
        else:
            pieces.append(comp)
    return dict(ah=ah, av=av, sel=sel, pieces=pieces, targets=targets)


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: min(pieces[i]))


def step(m, action):
    a = action.get('action_id') if isinstance(action, dict) else action
    sel, pieces = m['sel'], m['pieces']
    if a == 5:
        od = order(pieces)
        if sel == 'H':
            m['sel'] = 'V'
        elif sel == 'V':
            m['sel'] = od[0] if od else 'H'
        else:
            k = od.index(sel)
            m['sel'] = od[k + 1] if k + 1 < len(od) else 'H'
        return True
    if a not in DIRS:
        return False
    dr, dc = DIRS[a]
    if sel == 'H':
        if dc or not 0 <= m['ah'] + dr < N:
            return False
        m['ah'] += dr
        return True
    if sel == 'V':
        if dr or not 0 <= m['av'] + dc < N:
            return False
        m['av'] += dc
        return True
    if sel is None:
        return False
    moved = {(r + dr, c + dc) for r, c in pieces[sel]}
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
    if any(not (0 <= r < N and 0 <= c < N) for r, c in moved) or moved & others:
        return False
    pieces[sel] = moved
    return True


def mirrors(m):
    ah, av, pieces = m['ah'], m['av'], m['pieces']
    occ = set().union(*pieces) if pieces else set()
    vis = set()
    for r, c in occ:
        for q, ok in (((r, 2 * av - c), c != av), ((2 * ah - r, c), r != ah),
                      ((2 * ah - r, 2 * av - c), r != ah and c != av)):
            if ok and 0 <= q[0] < N and 0 <= q[1] < N and q not in occ:
                vis.add(q)
    return vis


def render(m, frame):
    out = [row[:] for row in frame]
    ah, av, sel = m['ah'], m['av'], m['sel']
    vis = mirrors(m)
    psel = set(m['pieces'][sel]) if isinstance(sel, int) else set()
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    for r in range(N):
        for c in range(N):
            b = (r, c)
            ring = centre = 9
            wall = r == ah or c == av
            if wall:
                ring = 10
                if (r == ah and c != av and sel == 'H') or (c == av and r != ah and sel == 'V') or \
                        (r == ah and c == av and sel in ('H', 'V')):
                    centre = 0
            if b in vis:
                ring = centre = 4
            if b in occ:
                ring = 5
                if b in psel:
                    centre = 0
            masked = b in occ or b in vis
            if b in m['targets']:
                centre = 11
                if not masked:
                    ring = 11
            for dy in range(3):
                for dx in range(3):
                    out[3 * r + dy][3 * c + dx] = centre if dy == dx == 1 else ring
    return out


def hud(frame, out, changed):
    col = [frame[y][63] for y in range(63)]
    if changed and 11 in col:
        out[col.index(11)][63] = 12


def transition_function(state, action, frame):
    m = parse(frame)
    changed = step(m, action)
    out = render(m, frame)
    hud(frame, out, changed)
    return out
