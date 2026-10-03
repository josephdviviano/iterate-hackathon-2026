# Mechanics: two mirror axes on a 3x3-block grid (H = row band at block row ah, V = column strip at block col av)
# plus holed pieces; A5 cycles selection H -> V -> pieces (by bbox min x, min y); A1/A2 move H or piece in x,
# A3/A4 move V or piece in y (blocked by bounds/other pieces; axes pass under pieces). Pieces mirror across V as
# gray reflections; mirrors across H / both are invisible erasers. A piece overlapping the H row gets an INVISIBLE
# V-mirror (step 268). Frame is composited per block then re-extracted by colour. Unconfirmed: piece cycle order.
N = 63
NB = 21
WALL, PIECE, REFL, TARGET, DOT = 10, 5, 4, 11, 0


def cells_of(state):
    g = {}
    for o in state:
        p = o.get('pixels')
        if not p:
            continue
        for i, row in enumerate(p):
            for j, v in enumerate(row):
                if v != -1:
                    g[(o['x'] + i, o['y'] + j)] = v
    return g


def block_groups(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in blocks:
                    blocks.discard(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def bkey(bs):
    return (min(b[0] for b in bs), min(b[1] for b in bs))


def parse(state):
    g = cells_of(state)
    wb = {(x // 3, y // 3) for (x, y), v in g.items() if v == WALL}
    rows = [sum(1 for b in wb if b[0] == r) for r in range(NB)]
    cols = [sum(1 for b in wb if b[1] == c) for c in range(NB)]
    ah, av = rows.index(max(rows)), cols.index(max(cols))
    pb = {(x // 3, y // 3) for (x, y), v in g.items() if v == PIECE}
    pieces = sorted(block_groups(pb), key=bkey)
    tb = {(x // 3, y // 3) for (x, y), v in g.items() if v == TARGET}
    zc = {b for b in {(x // 3, y // 3) for (x, y), v in g.items() if v == DOT}
          if g.get((b[0] * 3 + 1, b[1] * 3 + 1)) == DOT}
    sel = None
    if any(b[0] == ah and b[1] != av and b not in pb for b in zc):
        sel = 'H'
    elif any(b[1] == av and b[0] != ah and b not in pb for b in zc):
        sel = 'V'
    else:
        for i, p in enumerate(pieces):
            if p & zc:
                sel = i
                break
    return {'ah': ah, 'av': av, 'pieces': pieces, 'targets': tb, 'sel': sel}


def step(m, action):
    sel, pieces = m['sel'], m['pieces']
    if isinstance(action, dict):
        return m
    if action == 5:
        if sel == 'H':
            m['sel'] = 'V'
        elif sel == 'V':
            m['sel'] = 0 if pieces else 'H'
        elif isinstance(sel, int):
            m['sel'] = sel + 1 if sel + 1 < len(pieces) else 'H'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None or sel is None:
        return m
    if sel == 'H':
        if d[0] and 0 <= m['ah'] + d[0] < NB:
            m['ah'] += d[0]
    elif sel == 'V':
        if d[1] and 0 <= m['av'] + d[1] < NB:
            m['av'] += d[1]
    else:
        p = pieces[sel]
        moved = {(a + d[0], b + d[1]) for a, b in p}
        others = set().union(*[q for i, q in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
        if all(0 <= a < NB and 0 <= b < NB for a, b in moved) and not (moved & others):
            pieces[sel] = moved
    return m


def stacks(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    st = {}

    def add(b, spr):
        if 0 <= b[0] < NB and 0 <= b[1] < NB:
            st.setdefault(b, []).append(spr)
    for i, p in enumerate(m['pieces']):
        for b in p:
            add(b, ('P', PIECE, False, DOT if sel == i else None))
    for p in m['pieces']:
        on_h = any(b[0] == ah for b in p)
        for b in p:
            add((b[0], 2 * av - b[1]), ('I', -1, False, -1) if on_h else ('R', REFL, False, REFL))
    for p in m['pieces']:
        for b in p:
            add((2 * ah - b[0], b[1]), ('I', -1, False, -1))
            add((2 * ah - b[0], 2 * av - b[1]), ('I', -1, False, -1))
    order = {'P': 0, 'R': 1, 'I': 2}
    for b in st:
        st[b].sort(key=lambda s: order[s[0]])
    for b in m['targets']:
        add(b, ('T', TARGET, True, TARGET))
    for r in range(NB):
        add((r, av), ('W', WALL, sel == 'V', DOT if sel == 'V' else None))
    for c in range(NB):
        add((ah, c), ('W', WALL, sel == 'H', DOT if sel == 'H' else None))
    return st


def render(m):
    img, top = {}, {}
    for (bx, by), s in stacks(m).items():
        t = s[0]
        cen = None
        for spr in s:
            if spr[2]:
                cen = spr[3]
                break
        if cen is None:
            cen = t[3] if t[3] is not None else -1
        for i in range(3):
            for j in range(3):
                c = (bx * 3 + i, by * 3 + j)
                if c[0] >= N or c[1] >= N:
                    continue
                if (i, j) == (1, 1):
                    img[c] = cen
                    top[c] = t[0] if cen == t[3] or t[0] == 'T' else None
                else:
                    img[c] = TARGET if t[0] == 'T' else t[1]
                    top[c] = t[0]
    return img, top


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return sorted(out, key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))


def obj(name, typ, cells, img, layer, tags):
    x0, y0 = min(p[0] for p in cells), min(p[1] for p in cells)
    x1, y1 = max(p[0] for p in cells), max(p[1] for p in cells)
    pix = [[img[(x, y)] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
            'layer': layer, 'tags': tags, 'pixels': pix}


def extract(img, top):
    out = []
    zeros = {c for c, v in img.items() if v == DOT}
    for colour, typ in ((WALL, 'wall'), (PIECE, 'player')):
        own = {c for c, v in img.items() if v == colour}
        for k, comp in enumerate(comps(own | zeros)):
            if not comp & own:
                continue
            if typ == 'wall':
                w = max(p[0] for p in comp) - min(p[0] for p in comp)
                h = max(p[1] for p in comp) - min(p[1] for p in comp)
                v = w >= h
                out.append(obj('wall_%s_%d' % ('v' if v else 'h', k), 'wall', comp, img, 1,
                               ['axis', 'vertical' if v else 'horizontal']))
            else:
                out.append(obj('piece_%d' % k, 'player', comp, img, 4, ['movable', 'black']))
    vis = {c for c, v in img.items() if v == REFL}
    inv = {c for c, t in top.items() if t == 'I' and img[c] == -1}
    for k, comp in enumerate(comps(vis | inv)):
        cv = comp & vis
        if cv:
            out.append(obj('reflection_%d' % k, 'reflection', cv, img, 3, ['mirror', 'gray']))
    groups = [[min(p[0] for p in c), min(p[1] for p in c), max(p[0] for p in c), max(p[1] for p in c), c]
              for c in comps({c for c, v in img.items() if v == TARGET})]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                dx = max(0, b[0] - a[2], a[0] - b[2])
                dy = max(0, b[1] - a[3], a[1] - b[3])
                if max(dx, dy) <= 4:
                    groups[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] | b[4]]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: (g[0], g[1]))
    for k, gr in enumerate(groups):
        out.append(obj('target_%d' % k, 'target', gr[4], img, 2, ['goal', 'yellow']))
    return out


def transition_function(state, action):
    m = step(parse(state), action)
    img, top = render(m)
    out = [dict(o) for o in state if not o.get('pixels')]
    return out + extract(img, top)
