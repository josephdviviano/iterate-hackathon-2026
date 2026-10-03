# Mechanics: 3x3-block mirror game, x=row. Model = {H row-axis ah, V col-axis av, pieces (block sets),
# selection (H/V/piece, shown by 0 centres), target blocks}. A5 cycles H->V->pieces; A1/A2 move H or a
# piece by rows, A3/A4 move V or a piece by cols (bounds, pieces block pieces). Each piece has a visible
# gray mirror across V and invisible (erasing) mirrors across H and both; a piece touching the H row gets
# an invisible V-mirror instead. Frame = layered composite, then re-extracted by colour. Stateless.
import copy

N = 21
SIZE = 64
RING = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]


def frame_of(state):
    F = [[-1] * SIZE for _ in range(SIZE)]
    for o in state:
        px = o.get('pixels')
        if not px:
            continue
        for i, row in enumerate(px):
            for j, v in enumerate(row):
                if v != -1:
                    F[o['x'] + i][o['y'] + j] = v
    return F


def block_groups(blocks):
    blocks, groups = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        g = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in blocks:
                    blocks.remove(nb)
                    g.add(nb)
                    st.append(nb)
        groups.append(g)
    return groups


def parse(state):
    F = frame_of(state)
    ring = lambda b: [F[3 * b[0] + i][3 * b[1] + j] for i, j in RING]
    centre = lambda b: F[3 * b[0] + 1][3 * b[1] + 1]
    allb = [(r, c) for r in range(N) for c in range(N)]
    wallb = [b for b in allb if 10 in ring(b)]
    ah = max(range(N), key=lambda r: sum(1 for b in wallb if b[0] == r))
    av = max(range(N), key=lambda c: sum(1 for b in wallb if b[1] == c))
    pieces = block_groups([b for b in allb if 5 in ring(b)])
    pieces.sort(key=lambda g: (min(b[0] for b in g), min(b[1] for b in g)))
    targets = {b for b in allb if centre(b) == 11}
    zeros = [b for b in allb if centre(b) == 0]
    sel = None
    if any(b[0] == ah and b[1] != av and 5 not in ring(b) for b in zeros):
        sel = 'H'
    elif any(b[1] == av and b[0] != ah and 5 not in ring(b) for b in zeros):
        sel = 'V'
    else:
        for k, g in enumerate(pieces):
            if any(b in g for b in zeros):
                sel = k
                break
    if sel is None and zeros:
        sel = 'H'
    return {'ah': ah, 'av': av, 'pieces': pieces, 'targets': targets, 'sel': sel}


def step(m, action):
    m = copy.deepcopy(m)
    sel = m['sel']
    if action == 5:
        order = ['H', 'V'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return m
    if action not in (1, 2, 3, 4) or sel is None:
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if sel == 'H':
        if dr and 0 <= m['ah'] + dr < N:
            m['ah'] += dr
    elif sel == 'V':
        if dc and 0 <= m['av'] + dc < N:
            m['av'] += dc
    else:
        moved = {(r + dr, c + dc) for r, c in m['pieces'][sel]}
        others = set().union(*[g for k, g in enumerate(m['pieces']) if k != sel])
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
            m['pieces'][sel] = moved
    return m


# sprite: (owner, ring colour, centre colour, centre solid)
def render(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    stacks = {}
    add = lambda b, s: stacks.setdefault(b, []).append(s)
    inb = lambda b: 0 <= b[0] < N and 0 <= b[1] < N
    pblocks = set()
    for k, g in enumerate(m['pieces']):
        for b in g:
            add(b, ('player', 5, 0 if sel == k else -1, False))
            pblocks.add(b)
    vis, inv = set(), set()
    for g in m['pieces']:
        touch = any(r == ah for r, c in g)
        for r, c in g:
            (inv if touch else vis).add((r, 2 * av - c))
            inv.add((2 * ah - r, c))
            inv.add((2 * ah - r, 2 * av - c))
    for b in vis:
        if inb(b) and b not in pblocks:
            add(b, ('reflection', 4, 4, False))
    for b in inv - vis:
        if inb(b) and b not in pblocks:
            add(b, ('I', -1, -1, False))
    for b in m['targets']:
        add(b, ('target', 11, 11, True))
    for r in range(N):
        add((r, av), ('wall', 10, 0 if sel == 'V' else -1, sel == 'V'))
    for c in range(N):
        add((ah, c), ('wall', 10, 0 if sel == 'H' else -1, sel == 'H'))
    F = [[-1] * SIZE for _ in range(SIZE)]
    O = [[None] * SIZE for _ in range(SIZE)]
    for (r, c), st in stacks.items():
        top = st[0]
        for i, j in RING:
            F[3 * r + i][3 * c + j], O[3 * r + i][3 * c + j] = top[1], top[0]
        solid = next((s for s in st if s[3]), None)
        s = solid or top
        F[3 * r + 1][3 * c + 1], O[3 * r + 1][3 * c + 1] = s[2], s[0]
    return F, O


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        g = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells:
                    cells.remove(nb)
                    g.add(nb)
                    st.append(nb)
        out.append(g)
    return out


def bbox(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def make(name, typ, layer, tags, cells, F):
    r0, c0, r1, c1 = bbox(cells)
    px = [[F[r][c] if (r, c) in cells else -1 for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return {'name': name, 'type': typ, 'x': r0, 'y': c0, 'w': r1 - r0 + 1, 'h': c1 - c0 + 1,
            'layer': layer, 'tags': tags, 'pixels': px}


def extract(F, O):
    allc = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    out = []
    zeros = {p for p in allc if F[p[0]][p[1]] == 0}
    for col, typ in ((10, 'wall'), (5, 'player')):
        own = {p for p in allc if F[p[0]][p[1]] == col}
        cs = sorted(comps(own | zeros), key=lambda g: bbox(g)[:2])
        for k, g in enumerate(cs):
            if not (g & own):
                continue
            r0, c0, r1, c1 = bbox(g)
            if typ == 'wall':
                v = (r1 - r0) >= (c1 - c0)
                out.append(make('wall_%s_%d' % ('v' if v else 'h', k), 'wall', 1,
                                ['axis', 'vertical' if v else 'horizontal'], g, F))
            else:
                out.append(make('piece_%d' % k, 'player', 4, ['movable', 'black'], g, F))
    visc = {p for p in allc if F[p[0]][p[1]] == 4}
    invc = {p for p in allc if O[p[0]][p[1]] == 'I'}
    cs = sorted(comps(visc | invc), key=lambda g: bbox(g)[:2])
    for k, g in enumerate(cs):
        if g & visc:
            out.append(make('reflection_%d' % k, 'reflection', 3, ['mirror', 'gray'], g & visc, F))
    groups = comps({p for p in allc if F[p[0]][p[1]] == 11})
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = bbox(groups[a]), bbox(groups[b])
                gap = max(B[0] - A[2], A[0] - B[2], B[1] - A[3], A[1] - B[3])
                if gap <= 4:
                    groups[a] |= groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: bbox(g)[:2])
    for k, g in enumerate(groups):
        out.append(make('target_%d' % k, 'target', 2, ['goal', 'yellow'], g, F))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    m = step(parse(state), action)
    F, O = render(m)
    rest = [copy.deepcopy(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    return rest + extract(F, O)
