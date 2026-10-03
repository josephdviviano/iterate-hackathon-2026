# Mechanics: two-axis mirror game on a 21x21 grid of 3x3 blocks (x=row, pixels x-major). Walls = H axis row ah + V axis col av;
# pieces (colour-5 block groups) mirror across V as visible gray reflections, across H / H+V as invisible occluders.
# A piece touching the H axis row gets an INVISIBLE V-mirror instead of the visible one. A5 cycles H->V->pieces; A1-4 move selection.
# Frame rendered by per-block layer stacks then re-extracted: wall/player comps of own∪0 cells, refl ranks over vis+inv, targets merged by bbox gap.
# Unconfirmed: piece order for A5 beyond observed cases; pieces assumed never to touch each other (no merged-piece split).
N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
TARGET_GAP = 4


def frame(state):
    g = {}
    for o in state:
        p = o.get('pixels')
        if not p:
            continue
        for i, row in enumerate(p):
            for j, v in enumerate(row):
                if v != -1:
                    g[(o['x'] + i, o['y'] + j)] = (v, o['type'])
    return g


def comps(cells):
    cells = set(cells)
    out = []
    while cells:
        s = cells.pop()
        c, st = [s], [s]
        while st:
            x, y = st.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (x + d[0], y + d[1])
                if n in cells:
                    cells.remove(n)
                    c.append(n)
                    st.append(n)
        out.append(c)
    return out


def block_comps(blocks):
    return comps(blocks)


def parse(state):
    g = frame(state)
    wallb = {}
    for (x, y), (v, t) in g.items():
        if t == 'wall' and v == 10:
            wallb.setdefault((x // 3, y // 3), True)
    rowcnt, colcnt = {}, {}
    for (r, c) in wallb:
        rowcnt[r] = rowcnt.get(r, 0) + 1
        colcnt[c] = colcnt.get(c, 0) + 1
    ah = max(rowcnt, key=lambda r: rowcnt[r])
    av = max(colcnt, key=lambda c: colcnt[c])
    pblocks = {(x // 3, y // 3) for (x, y), (v, t) in g.items()
               if v == 5 and x % 3 == 0 and y % 3 == 0}
    pieces = [frozenset(c) for c in block_comps(pblocks)]
    targets = {(x // 3, y // 3) for (x, y), (v, t) in g.items() if v == 11}
    zero = [(x // 3, y // 3) for (x, y), (v, t) in g.items()
            if v == 0 and x % 3 == 1 and y % 3 == 1]
    sel = None
    if any(r == ah and c != av and (r, c) not in pblocks for r, c in zero):
        sel = 'H'
    elif any(c == av and (r, c) not in pblocks for r, c in zero):
        sel = 'V'
    else:
        for i, p in enumerate(pieces):
            if any(b in p for b in zero):
                sel = i
    others = [o for o in state if o['type'] == 'counter']
    return {'ah': ah, 'av': av, 'pieces': pieces, 'targets': targets, 'sel': sel, 'others': others}


def order_pieces(pieces):
    return sorted(range(len(pieces)), key=lambda i: (min(b[0] for b in pieces[i]), min(b[1] for b in pieces[i])))


def step(m, action):
    a = action['action_id'] if isinstance(action, dict) else action
    pieces = list(m['pieces'])
    sel = m['sel']
    if a == 5:
        order = order_pieces(pieces)
        if sel == 'H':
            sel = 'V'
        elif sel == 'V':
            sel = order[0] if order else 'H'
        elif sel is None:
            sel = 'H'
        else:
            k = order.index(sel)
            sel = order[k + 1] if k + 1 < len(order) else 'H'
    elif a in DIRS:
        dr, dc = DIRS[a]
        if sel == 'H' and dc == 0 and 0 <= m['ah'] + dr < N:
            m['ah'] += dr
        elif sel == 'V' and dr == 0 and 0 <= m['av'] + dc < N:
            m['av'] += dc
        elif isinstance(sel, int):
            nb = frozenset((r + dr, c + dc) for r, c in pieces[sel])
            occ = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
            if all(0 <= r < N and 0 <= c < N for r, c in nb) and not (nb & occ):
                pieces[sel] = nb
    m['pieces'] = pieces
    m['sel'] = sel
    return m


def render(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    stacks = {}

    def push(b, spr):
        stacks.setdefault(b, []).append(spr)
    # sprite = (ring value, centre solid?, centre value); -2 = invisible occluder
    for i, p in enumerate(m['pieces']):
        for b in p:
            push(b, ('P', 5, False, 0 if sel == i else -1))
    vis, inv = set(), set()
    for p in m['pieces']:
        on_h = any(r == ah for r, c in p)
        for r, c in p:
            for (rr, cc), visible in (((r, 2 * av - c), not on_h), ((2 * ah - r, c), False), ((2 * ah - r, 2 * av - c), False)):
                if 0 <= rr < N and 0 <= cc < N and (rr, cc) not in allp:
                    (vis if visible else inv).add((rr, cc))
    for b in vis:
        push(b, ('R', 4, False, 4))
    for b in inv - vis:
        push(b, ('I', -2, False, -2))
    for b in m['targets']:
        push(b, ('T', 11, True, 11))
    for r in range(N):
        push((r, av), ('W', 10, sel == 'V', 0 if sel == 'V' else -1))
    for c in range(N):
        push((ah, c), ('W', 10, sel == 'H', 0 if sel == 'H' else -1))
    g = {}
    for (r, c), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    g[(3 * r + i, 3 * c + j)] = top[1]
        solid = [s for s in st if s[2]]
        g[(3 * r + 1, 3 * c + 1)] = solid[0][3] if solid else top[3]
    return {k: v for k, v in g.items() if v != -1}


def mk(name, typ, tags, layer, cells, g):
    xs = [x for x, y in cells]
    ys = [y for x, y in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for x, y in cells:
        pix[x - x0][y - y0] = g[(x, y)]
    return {'name': name, 'type': typ, 'tags': tags, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer, 'pixels': pix}


def bbox(c):
    return (min(x for x, y in c), min(y for x, y in c))


def extract(g, others):
    out = [dict(o) for o in others]
    zeros = [k for k, v in g.items() if v == 0]
    for val, typ, layer in ((10, 'wall', 1), (5, 'player', 4)):
        cs = sorted(comps([k for k, v in g.items() if v == val] + zeros), key=bbox)
        for i, c in enumerate(cs):
            if not any(g[k] == val for k in c):
                continue
            o = mk('', typ, None, layer, c, g)
            if typ == 'wall':
                v = o['w'] >= o['h']
                o['name'] = 'wall_%s_%d' % ('v' if v else 'h', i)
                o['tags'] = ['axis', 'vertical' if v else 'horizontal']
            else:
                o['name'] = 'piece_%d' % i
                o['tags'] = ['movable', 'black']
            out.append(o)
    cs = sorted(comps([k for k, v in g.items() if v in (4, -2)]), key=bbox)
    for i, c in enumerate(cs):
        vc = [k for k in c if g[k] == 4]
        if vc:
            out.append(mk('reflection_%d' % i, 'reflection', ['mirror', 'gray'], 3, vc, g))
    cl = [[min(x for x, y in c), min(y for x, y in c), max(x for x, y in c), max(y for x, y in c), c]
          for c in comps([k for k, v in g.items() if v == 11])]
    merged = True
    while merged:
        merged = False
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                gap = max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])
                if gap <= TARGET_GAP:
                    cl[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    del cl[j]
                    merged = True
                    break
            if merged:
                break
    cl.sort(key=lambda t: (t[0], t[1]))
    for i, t in enumerate(cl):
        out.append(mk('target_%d' % i, 'target', ['goal', 'yellow'], 2, t[4], g))
    return out


def transition_function(state, action):
    m = step(parse(state), action)
    return extract(render(m), m['others'])
