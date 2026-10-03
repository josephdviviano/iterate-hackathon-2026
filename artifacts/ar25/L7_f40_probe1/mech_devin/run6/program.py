# Mechanics: two-axis mirror board on 21x21 cells of 3x3 px (x=row). Row axis R and column axis C are wall bands;
# A5 cycles selection R->C->pieces; A1/A2 move selection x-/+3, A3/A4 y-/+3. Pieces reflect across C into visible
# solid grey cells; a piece touching row R has its C-mirror hidden; R- and RC-mirrors are always hidden. Hidden cells
# erase wall rings and (like pieces/visible mirrors) mask targets to their centre pixel. Frame is re-rendered and
# re-extracted (4-conn comps, foreign 0-dots/hidden comps consume name indices). Unconfirmed: piece order, blocking.
N = 21


def parse(state):
    wallpx, zeros, pblocks, tblocks = [], set(), set(), set()
    for o in state:
        P = o.get('pixels')
        if not P:
            continue
        for i, row in enumerate(P):
            for j, v in enumerate(row):
                if v < 0:
                    continue
                X, Y = o['x'] + i, o['y'] + j
                if v == 0:
                    zeros.add((X, Y))
                if o['type'] == 'wall':
                    wallpx.append((X, Y))
                elif o['type'] == 'player' and v == 5:
                    pblocks.add((X // 3, Y // 3))
                elif o['type'] == 'target':
                    tblocks.add((X // 3, Y // 3))
    rc, cc = [0] * N, [0] * N
    for X, Y in wallpx:
        if X < 63 and Y < 63:
            rc[X // 3] += 1
            cc[Y // 3] += 1
    ax, ay = rc.index(max(rc)), cc.index(max(cc))
    pieces = comps(pblocks)
    pieces.sort(key=lambda p: min(p))
    sel = None
    dots = {(X // 3, Y // 3) for X, Y in zeros}
    for i, p in enumerate(pieces):
        if any(b in dots and b[0] != ax and b[1] != ay for b in p):
            sel = i
    if sel is None:
        if any(b[0] == ax and b[1] != ay for b in dots):
            sel = 'R'
        elif any(b[1] == ay and b[0] != ax for b in dots):
            sel = 'C'
    return {'ax': ax, 'ay': ay, 'sel': sel, 'pieces': pieces, 'targets': tblocks}


def comps(cells, conn=((1, 0), (-1, 0), (0, 1), (0, -1))):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        c = set(st)
        while st:
            a, b = st.pop()
            for da, db in conn:
                n = (a + da, b + db)
                if n in cells:
                    cells.remove(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def on_board(b):
    return 0 <= b[0] < N and 0 <= b[1] < N


def mirrors(m):
    ax, ay = m['ax'], m['ay']
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    vis, hid = set(), set()
    for p in m['pieces']:
        touches = any(r == ax for r, c in p)
        for r, c in p:
            cand = []
            if c != ay:
                cand.append(((r, 2 * ay - c), touches))
            if r != ax:
                cand.append(((2 * ax - r, c), True))
                if c != ay:
                    cand.append(((2 * ax - r, 2 * ay - c), True))
            for d, h in cand:
                if on_board(d) and d not in allp:
                    (hid if h else vis).add(d)
    return vis, hid - vis


def cellpx(b):
    return [(3 * b[0] + i, 3 * b[1] + j) for i in range(3) for j in range(3)]


def centre(b):
    return (3 * b[0] + 1, 3 * b[1] + 1)


def render(m):
    ax, ay, sel = m['ax'], m['ay'], m['sel']
    vis, hid = mirrors(m)
    G = {}
    for k in range(N):
        for b in {(ax, k), (k, ay)}:
            if b in hid:
                continue
            for p in cellpx(b):
                G[p] = (10, 'wall')
            dot = (b[0] == ax and sel == 'R') or (b[1] == ay and sel == 'C')
            if dot:
                G[centre(b)] = (0, 'wall')
            else:
                G.pop(centre(b))
    for b in vis:
        for p in cellpx(b):
            G[p] = (4, 'reflection')
    allp = set()
    for i, pc in enumerate(m['pieces']):
        allp |= pc
        for b in pc:
            for p in cellpx(b):
                G[p] = (5, 'player')
            c = centre(b)
            if sel == i:
                G[c] = (0, 'player')
            else:
                G.pop(c)
                if lower_dot(m, b):
                    G[c] = (0, 'player')
    mask = allp | vis | hid
    tparts = []
    for b in m['targets']:
        px = [centre(b)] if b in mask else cellpx(b)
        for p in px:
            G[p] = (11, 'target')
        tparts.append(set(px))
    return G, hid, tparts


def lower_dot(m, b):
    ax, ay, sel = m['ax'], m['ay'], m['sel']
    _, hid = mirrors(m)
    if b in hid:
        return False
    return (b[0] == ax and sel == 'R') or (b[1] == ay and sel == 'C')


def bbox(px):
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    return min(xs), min(ys), max(xs), max(ys)


def mkobj(name, typ, tags, layer, px, G):
    x0, y0, x1, y1 = bbox(px)
    pix = [[-1] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    for p in px:
        pix[p[0] - x0][p[1] - y0] = G[p][0]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': pix}


def pixcomps(px):
    return comps(px)


def extract(m, scenery):
    G, hid, tparts = render(m)
    ax, ay = m['ax'], m['ay']
    out = list(scenery)
    zeros = {p for p, v in G.items() if v[0] == 0}
    # walls and pieces: comps of own pixels U all 0 pixels, dot-only comps consume indices
    for typ, colour in (('wall', 10), ('player', 5)):
        own = {p for p, v in G.items() if v[1] == typ}
        cs = pixcomps(own | zeros)
        cs.sort(key=lambda c: bbox(c)[:2])
        for i, c in enumerate(cs):
            if not any(G[p][0] == colour for p in c):
                continue
            c = {p for p in c if G[p][1] == typ or G[p][0] == 0}
            if typ == 'wall':
                x0, y0, x1, y1 = bbox(c)
                v = x1 - x0 >= y1 - y0
                out.append(mkobj('wall_%s_%d' % ('v' if v else 'h', i), 'wall',
                                 ['axis', 'vertical' if v else 'horizontal'], 1, c, G))
            else:
                out.append(mkobj('piece_%d' % i, 'player', ['movable', 'black'], 4, c, G))
    # reflections: visible comps ranked together with hidden-cell comps
    vis = {p for p, v in G.items() if v[0] == 4}
    items = [(bbox(c)[:2], c) for c in pixcomps(vis)]
    items += [((3 * min(c)[0], 3 * min(c)[1]), None) for c in comps(hid)]
    items.sort(key=lambda t: t[0])
    for i, (k, c) in enumerate(items):
        if c is not None:
            out.append(mkobj('reflection_%d' % i, 'reflection', ['mirror', 'gray'], 3, c, G))
    # targets: merge parts while bbox gap <= 4
    boxes = [(bbox(p), p) for p in tparts]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][0], boxes[j][0]
                if max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3]) <= 4:
                    u = boxes[i][1] | boxes[j][1]
                    boxes[i] = (bbox(u), u)
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    boxes.sort(key=lambda t: t[0][:2])
    for i, (bb, p) in enumerate(boxes):
        out.append(mkobj('target_%d' % i, 'target', ['goal', 'yellow'], 2, p, G))
    return out


def step(m, action):
    sel, n = m['sel'], len(m['pieces'])
    if action == 5:
        order = ['R', 'C'] + list(range(n))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'R'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None or sel is None:
        return m
    if sel == 'R':
        if 0 <= m['ax'] + d[0] < N:
            m['ax'] += d[0]
    elif sel == 'C':
        if 0 <= m['ay'] + d[1] < N:
            m['ay'] += d[1]
    else:
        moved = {(r + d[0], c + d[1]) for r, c in m['pieces'][sel]}
        others = set().union(set(), *[p for i, p in enumerate(m['pieces']) if i != sel])
        if all(on_board(b) for b in moved) and not (moved & others):
            m['pieces'][sel] = moved
    return m


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    scenery = [o for o in state if o['type'] == 'counter']
    m = parse(state)
    return extract(step(m, action), scenery)
