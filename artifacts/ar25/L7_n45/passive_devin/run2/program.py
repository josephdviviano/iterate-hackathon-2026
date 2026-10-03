# Mechanics: two-axis mirror puzzle on a 21x21 grid of 3x3 blocks. Model = {H-axis block row, V-axis block col,
# pieces (block sets), target blocks, selection}; ACTION5 cycles H -> V -> pieces (by min block) -> H; A1/A2 move H or a
# piece by a block row, A3/A4 move V or a piece by a block col (pieces blocked by bounds/other pieces; axes by bounds).
# Pieces reflect across V (visible gray) and across H / H+V (invisible occluders); frame is re-rendered by layer
# and re-extracted (4-conn comps, names ranked by bbox incl. foreign 0 dots). Unconfirmed: ACTION6/7 (no-op), piece order.
N = 21
SZ = 63
WALL, PIECE, REFL, TGT, ZERO = 10, 5, 4, 11, 0


def _cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def parse(state):
    wall = {}
    pieces, tblocks = [], set()
    for o in state:
        t = o['type']
        if t == 'wall':
            for x, y, v in _cells(o):
                wall[(x, y)] = v
        elif t == 'player':
            cs = list(_cells(o))
            blocks = {(x // 3, y // 3) for x, y, v in cs if v == PIECE}
            zeros = {(x // 3, y // 3) for x, y, v in cs if v == ZERO and x % 3 == 1 and y % 3 == 1}
            if blocks:
                pieces.append((blocks, zeros & blocks))
        elif t == 'target':
            for x, y, v in _cells(o):
                tblocks.add((x // 3, y // 3))
    rows, cols = [set() for _ in range(N)], [set() for _ in range(N)]
    for (x, y) in wall:
        rows[x // 3].add(y // 3)
        cols[y // 3].add(x // 3)
    ah = max(range(N), key=lambda r: len(rows[r]))
    av = max(range(N), key=lambda c: len(cols[c]))
    pblocks = set().union(*[p[0] for p in pieces]) if pieces else set()

    def zero_at(r, c):
        return wall.get((3 * r + 1, 3 * c + 1)) == ZERO and (r, c) not in pblocks
    sel = None
    if any(zero_at(ah, c) for c in range(N) if c != av):
        sel = 'H'
    elif any(zero_at(r, av) for r in range(N) if r != ah):
        sel = 'V'
    elif zero_at(ah, av):
        sel = 'H'
    else:
        for i, (b, z) in enumerate(sorted(pieces, key=lambda p: min(p[0]))):
            if z:
                sel = i
    plist = [p[0] for p in sorted(pieces, key=lambda p: min(p[0]))]
    return {'ah': ah, 'av': av, 'pieces': plist, 'targets': tblocks, 'sel': sel}


MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def step(m, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    sel = m['sel']
    if aid == 5:
        order = ['H', 'V'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return m
    if aid not in MOVES:
        return m
    dr, dc = MOVES[aid]
    if sel == 'H' and dr:
        if 0 <= m['ah'] + dr < N:
            m['ah'] += dr
    elif sel == 'V' and dc:
        if 0 <= m['av'] + dc < N:
            m['av'] += dc
    elif isinstance(sel, int):
        moved = {(r + dr, c + dc) for r, c in m['pieces'][sel]}
        others = set().union(*[p for i, p in enumerate(m['pieces']) if i != sel]) if len(m['pieces']) > 1 else set()
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
            m['pieces'][sel] = moved
    return m


def render(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    stacks = {}

    def push(b, kind, s=False):
        if 0 <= b[0] < N and 0 <= b[1] < N:
            stacks.setdefault(b, []).append((kind, s))
    pb = set()
    for i, p in enumerate(m['pieces']):
        for b in p:
            push(b, 'P', sel == i)
            pb.add(b)
    allp = [b for p in m['pieces'] for b in p]
    vis = {(r, 2 * av - c) for r, c in allp} - pb
    inv = ({(2 * ah - r, c) for r, c in allp} | {(2 * ah - r, 2 * av - c) for r, c in allp}) - pb - vis
    for b in vis:
        push(b, 'R')
    for b in inv:
        push(b, 'I')
    for b in m['targets']:
        push(b, 'T')
    for r in range(N):
        push((r, av), 'W', sel == 'V')
    for c in range(N):
        push((ah, c), 'W', sel == 'H')
    col, invc = {}, set()
    ring = {'P': PIECE, 'R': REFL, 'T': TGT, 'W': WALL}
    for (r, c), st in stacks.items():
        top = st[0][0]
        for i in range(3):
            for j in range(3):
                x, y = 3 * r + i, 3 * c + j
                if (i, j) != (1, 1):
                    if top == 'I':
                        invc.add((x, y))
                    else:
                        col[(x, y)] = ring[top]
                    continue
                v = None
                for k, s in st:
                    if k == 'T':
                        v = TGT
                        break
                    if k == 'W' and s:
                        v = ZERO
                        break
                if v is None:
                    k, s = st[0]
                    v = {'P': ZERO if s else None, 'R': REFL, 'I': None, 'W': None}[k]
                    if k == 'I':
                        invc.add((x, y))
                if v is not None:
                    col[(x, y)] = v
    return col, invc


def comps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, todo = {s}, [s]
        while todo:
            x, y = todo.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    todo.append(n)
        out.append(comp)
    return out


def bbox(cs):
    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
    return min(xs), min(ys), max(xs), max(ys)


def mkobj(name, typ, layer, tags, cs, col):
    x0, y0, x1, y1 = bbox(cs)
    pix = [[col[(x, y)] if (x, y) in cs else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
            'layer': layer, 'tags': tags, 'pixels': pix}


def ranked(cells):
    return list(enumerate(sorted(comps(cells), key=lambda c: bbox(c)[:2])))


def extract(col, invc):
    out = []
    zeros = {p for p, v in col.items() if v == ZERO}
    for i, c in ranked({p for p, v in col.items() if v == WALL} | zeros):
        if not any(col[p] == WALL for p in c):
            continue
        x0, y0, x1, y1 = bbox(c)
        hz = (y1 - y0) > (x1 - x0)
        out.append(mkobj('wall_%s_%d' % ('h' if hz else 'v', i), 'wall', 1,
                         ['axis', 'horizontal' if hz else 'vertical'], c, col))
    for i, c in ranked({p for p, v in col.items() if v == PIECE} | zeros):
        if any(col[p] == PIECE for p in c):
            out.append(mkobj('piece_%d' % i, 'player', 4, ['movable', 'black'], c, col))
    for i, c in ranked({p for p, v in col.items() if v == REFL} | invc):
        vis = {p for p in c if col.get(p) == REFL}
        if vis:
            out.append(mkobj('reflection_%d' % i, 'reflection', 3, ['mirror', 'gray'], vis, col))
    groups = [bbox(c) + (c,) for c in comps({p for p, v in col.items() if v == TGT})]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = groups[a], groups[b]
                gap = max(A[0] - B[2], B[0] - A[2], A[1] - B[3], B[1] - A[3])
                if gap <= 4:
                    cs = A[4] | B[4]
                    groups[a] = bbox(cs) + (cs,)
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    for i, g in enumerate(sorted(groups, key=lambda g: (g[0], g[1]))):
        out.append(mkobj('target_%d' % i, 'target', 2, ['goal', 'yellow'], g[4], col))
    return out


def transition_function(state, action):
    m = step(parse(state), action)
    col, invc = render(m)
    hud = [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    return hud + extract(col, invc)
