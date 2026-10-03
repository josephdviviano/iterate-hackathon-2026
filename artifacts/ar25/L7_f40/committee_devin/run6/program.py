# Mechanics: two mirror axes (h wall row R, v wall col C, on a 21x21 grid of 3x3 blocks) + holed pieces + static targets.
# ACTION5 cycles selection h -> v -> pieces (by bbox x,y) -> h; A1/A2 move the selection a block up/down, A3/A4 left/right
# (h ignores A3/A4, v ignores A1/A2; pieces stay in bounds and off other pieces). Pieces reflect across v (visible gray),
# across h and both (invisible occluders). Frame = layered composite, re-extracted as 4-conn components; names rank by (x,y).
# Unconfirmed: piece cycle order (first by x,y used), piece-vs-piece blocking, clicks/ACTION7 (treated as no-ops).
N = 21
INV = -2


def _blk(r, c):
    return r // 3, c // 3


def parse(state):
    comp, owner = {}, {}
    pieces = []
    other = []
    for o in state:
        if 'pixels' not in o:
            other.append(o)
            continue
        if o['type'] == 'player':
            pieces.append(set())
        for i, row in enumerate(o['pixels']):
            for j, v in enumerate(row):
                if v == -1:
                    continue
                r, c = o['x'] + i, o['y'] + j
                comp[(r, c)] = v
                owner[(r, c)] = o['type']
                if o['type'] == 'player' and v == 5:
                    pieces[-1].add(_blk(r, c))
    targets = {_blk(r, c) for (r, c), v in comp.items() if v == 11}
    rows, cols = {}, {}
    for (r, c), v in comp.items():
        if v == 10:
            b = _blk(r, c)
            rows.setdefault(b[0], set()).add(b[1])
            cols.setdefault(b[1], set()).add(b[0])
    R = max(rows, key=lambda k: len(rows[k])) if rows else None
    C = max(cols, key=lambda k: len(cols[k])) if cols else None

    def zero_wall(br, bc):
        p = (3 * br + 1, 3 * bc + 1)
        return comp.get(p) == 0 and owner.get(p) == 'wall'
    sel = None
    if R is not None and any(zero_wall(R, c) for c in range(N) if c != C):
        sel = 'h'
    elif C is not None and any(zero_wall(r, C) for r in range(N) if r != R):
        sel = 'v'
    else:
        cands = []
        for k, p in enumerate(pieces):
            zs = [b for b in p if comp.get((3 * b[0] + 1, 3 * b[1] + 1)) == 0]
            if zs:
                cands.append((0 if any(b[0] != R and b[1] != C for b in zs) else 1, k))
        if cands:
            sel = min(cands)[1]
    return dict(R=R, C=C, sel=sel, pieces=pieces, targets=targets, other=other)


def order(pieces):
    return sorted(range(len(pieces)), key=lambda k: min(pieces[k]))


def step(m, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    sel, pieces = m['sel'], m['pieces']
    if aid == 5:
        cyc = ['h', 'v'] + order(pieces)
        m['sel'] = cyc[(cyc.index(sel) + 1) % len(cyc)] if sel in cyc else 'h'
        return m
    if aid not in (1, 2, 3, 4):
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[aid]
    if sel == 'h':
        if dr and 0 <= m['R'] + dr < N:
            m['R'] += dr
    elif sel == 'v':
        if dc and 0 <= m['C'] + dc < N:
            m['C'] += dc
    elif sel is not None:
        nb = {(r + dr, c + dc) for r, c in pieces[sel]}
        others = set().union(*[p for k, p in enumerate(pieces) if k != sel]) if len(pieces) > 1 else set()
        if all(0 <= r < N and 0 <= c < N for r, c in nb) and not (nb & others):
            pieces[sel] = nb
    return m


def render(m):
    R, C, sel, pieces = m['R'], m['C'], m['sel'], m['pieces']
    vref, iref = set(), set()
    for p in pieces:
        for r, c in p:
            if C is not None:
                vref.add((r, 2 * C - c))
            if R is not None:
                iref.add((2 * R - r, c))
            if R is not None and C is not None:
                iref.add((2 * R - r, 2 * C - c))
    inb = lambda b: 0 <= b[0] < N and 0 <= b[1] < N
    vref = {b for b in vref if inb(b)}
    iref = {b for b in iref if inb(b)}
    cells = {}
    for br in range(N):
        for bc in range(N):
            b = (br, bc)
            # stack top->bottom: (owner, frame colour, centre: ('solid', v) | ('hole', fill))
            st = []
            for k, p in enumerate(pieces):
                if b in p:
                    st.append(('piece', 5, ('hole', 0 if sel == k else None)))
            if b in vref:
                st.append(('vref', 4, ('hole', 4)))
            if b in iref:
                st.append(('iref', INV, ('hole', None)))
            if b in m['targets']:
                st.append(('target', 11, ('solid', 11)))
            if br == R or bc == C:
                on = (br == R and sel == 'h') or (bc == C and sel == 'v')
                st.append(('wall', 10, ('solid', 0) if on else ('hole', None)))
            if not st:
                continue
            for i in range(3):
                for j in range(3):
                    p = (3 * br + i, 3 * bc + j)
                    if (i, j) != (1, 1):
                        cells[p] = (st[0][1], st[0][0])
                        continue
                    solid = next((s for s in st if s[2][0] == 'solid'), None)
                    if solid:
                        v = solid[2][1]
                        cells[p] = (v, st[0][0] if v == 0 else solid[0])
                    else:
                        if st[0][2][1] is not None:
                            cells[p] = (st[0][2][1], st[0][0])
    return cells


def components(cells):
    seen, out = set(), []
    for p in sorted(cells):
        if p in seen:
            continue
        stack, comp = [p], []
        seen.add(p)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in cells and q not in seen:
                    seen.add(q)
                    stack.append(q)
        out.append(comp)
    return out


def bbox(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), min(cs), max(rs), max(cs)


def merge_targets(comps, gap=3):
    groups = [list(c) for c in comps]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = bbox(groups[i]), bbox(groups[j])
                g = max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3]) - 1
                if g <= gap:
                    groups[i] += groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups


def make(comp, cells, typ, name, tags, layer):
    x0, y0, x1, y1 = bbox(comp)
    px = [[-1] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    for r, c in comp:
        px[r - x0][c - y0] = cells[r, c][0]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': px}


def ranked(items, extra):
    keys = sorted([(bbox(c)[:2], 0, i) for i, c in enumerate(items)] + [(p, 1, -1) for p in extra])
    return {k[2]: n for n, k in enumerate(keys) if k[1] == 0}


def extract(cells, other):
    by = {}
    for p, (v, o) in cells.items():
        by.setdefault(o, {})[p] = cells[p]
    comps = {o: components(d) for o, d in by.items()}
    zeros = lambda o: [p for p, (v, w) in cells.items() if v == 0 and w == o]
    out = [dict(o) for o in other]
    walls = comps.get('wall', [])
    for i, n in ranked(walls, zeros('piece')).items():
        x0, y0, x1, y1 = bbox(walls[i])
        hz = (y1 - y0) > (x1 - x0)
        out.append(make(walls[i], cells, 'wall', ('wall_h_' if hz else 'wall_v_') + str(n),
                        ['axis', 'horizontal' if hz else 'vertical'], 1))
    ps = comps.get('piece', [])
    for i, n in ranked(ps, zeros('wall')).items():
        out.append(make(ps[i], cells, 'player', 'piece_%d' % n, ['movable', 'black'], 4))
    vr, ir = comps.get('vref', []), comps.get('iref', [])
    for i, n in ranked(vr + ir, []).items():
        if i < len(vr):
            out.append(make(vr[i], cells, 'reflection', 'reflection_%d' % n, ['mirror', 'gray'], 3))
    ts = merge_targets(comps.get('target', []))
    for i, n in ranked(ts, []).items():
        out.append(make(ts[i], cells, 'target', 'target_%d' % n, ['goal', 'yellow'], 2))
    return out


def transition_function(state, action):
    m = parse(state)
    m = step(m, action)
    return extract(render(m), m['other'])
