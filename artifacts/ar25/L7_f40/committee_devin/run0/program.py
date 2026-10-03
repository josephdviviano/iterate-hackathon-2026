# Mechanics: h-axis row band + v-axis column band (walls) and 2 pieces on a 21x21 grid of 3x3 blocks; A5 cycles selection h->v->pieces by (x,y).
# A1/A2 move the selected h-axis/piece x-/+3, A3/A4 the v-axis/piece y-/+3 (in bounds, pieces not overlapping); else no-op. Selection = 0 hole fills.
# Reflections: v-mirror of piece blocks = gray blocks (hole fill 4); h- and hv-mirrors are invisible erasers (centre shows target below); none on pieces.
# Render wall<target<reflection<piece, holes show what is below else fill; re-extract 4-conn comps, wall tag by bbox aspect, targets merged at bbox gap<=3.
# Names rank by (x,y); wall/piece ranks also count 0 pixels owned by the other type, invisible comps count for reflections. Unconfirmed: blocking rules, A6/A7.
import json

N = 21
S = 63
INV_FILL = -1
_mem = {'out': None, 'targets': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def parse(state):
    wall_rows, wall_cols, wall_zero = {}, {}, []
    pieces, targets, other0 = [], set(), []
    for o in state:
        px = o.get('pixels')
        if px is None:
            continue
        for i, row in enumerate(px):
            for j, v in enumerate(row):
                if v == -1:
                    continue
                x, y = o['x'] + i, o['y'] + j
                if o['type'] == 'wall':
                    wall_rows[x // 3] = wall_rows.get(x // 3, 0) + 1
                    wall_cols[y // 3] = wall_cols.get(y // 3, 0) + 1
                    if v == 0:
                        wall_zero.append((x, y))
                elif o['type'] == 'target' and v == 11:
                    targets.add((x // 3, y // 3))
    for o in state:
        if o['type'] != 'player':
            continue
        blocks, zeros = set(), []
        for i, row in enumerate(o['pixels']):
            for j, v in enumerate(row):
                x, y = o['x'] + i, o['y'] + j
                if v == 5 and x % 3 == 0 and y % 3 == 0:
                    blocks.add((x // 3, y // 3))
                if v == 0:
                    zeros.append((x, y))
        pieces.append([blocks, zeros])
    ha = max(wall_rows, key=lambda k: wall_rows[k])
    vb = max(wall_cols, key=lambda k: wall_cols[k])
    sel = None
    if any(x == 3 * ha + 1 and y != 3 * vb + 1 for x, y in wall_zero):
        sel = 'h'
    elif any(y == 3 * vb + 1 and x != 3 * ha + 1 for x, y in wall_zero):
        sel = 'v'
    pieces.sort(key=lambda p: min((b[0], b[1]) for b in p[0]))
    if sel is None:
        for k, (bl, zs) in enumerate(pieces):
            if zs:
                sel = k
                break
    if sel is None:
        sel = 'h'
    return {'h': ha, 'v': vb, 'sel': sel, 'pieces': [p[0] for p in pieces], 'targets': targets}


def step(m, action):
    a = action['action_id'] if isinstance(action, dict) else action
    sel = m['sel']
    if a == 5:
        order = ['h', 'v'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(a)
    if d is None:
        return m
    dx, dy = d
    if sel == 'h':
        if dx and 0 <= m['h'] + dx < N:
            m['h'] += dx
    elif sel == 'v':
        if dy and 0 <= m['v'] + dy < N:
            m['v'] += dy
    else:
        nb = {(i + dx, j + dy) for i, j in m['pieces'][sel]}
        others = set().union(*[p for k, p in enumerate(m['pieces']) if k != sel]) if len(m['pieces']) > 1 else set()
        if all(0 <= i < N and 0 <= j < N for i, j in nb) and not (nb & others):
            m['pieces'][sel] = nb
    return m


def reflections(m):
    occ = set().union(*m['pieces'])
    A, B = m['h'], m['v']
    vis, inv = set(), set()
    for i, j in occ:
        for (ri, rj), dst in (((i, 2 * B - j), vis), ((2 * A - i, j), inv), ((2 * A - i, 2 * B - j), inv)):
            if 0 <= ri < N and 0 <= rj < N and (ri, rj) not in occ:
                dst.add((ri, rj))
    inv -= vis
    return vis, inv


def render(m):
    """Return cell map {(x,y): (value, owner)}; owner = ('wall',kind)|('piece',k)|('refl',)|('target',)."""
    vis, inv = reflections(m)
    sel = m['sel']
    grid = {}
    # sprites top-first: (solid value | None for hole | 'erase', hole fill, owner)
    def sprites_at(x, y):
        bi, bj = x // 3, y // 3
        centre = (x % 3 == 1 and y % 3 == 1)
        out = []
        for k, p in enumerate(m['pieces']):
            if (bi, bj) in p:
                out.append((None if centre else 5, 0 if sel == k else -1, ('piece', k)))
        if (bi, bj) in vis:
            out.append((None if centre else 4, 4, ('refl',)))
        if (bi, bj) in inv:
            out.append((None if centre else 'erase', INV_FILL, ('refl',)))
        if (bi, bj) in m['targets']:
            out.append((11, 11, ('target',)))
        hw = bi == m['h']
        vw = bj == m['v']
        if hw or vw:
            hole = centre and ((hw and y % 3 == 1) or (vw and x % 3 == 1))
            kind = 'h' if hw and (not vw or x % 3 != 1 or not centre) else 'v'
            if hw and vw:
                kind = 'x'
            out.append((None if hole else 10, 0 if sel in ('h', 'v') and ((sel == 'h' and hw) or (sel == 'v' and vw)) else -1, ('wall', kind)))
        return out
    for x in range(S):
        for y in range(S):
            sp = sprites_at(x, y)
            if not sp:
                continue
            first_hole = None
            val = None
            for solid, fill, owner in sp:
                if solid == 'erase':
                    val = (-1, None)
                    break
                if solid is not None:
                    val = (solid, owner)
                    break
                if first_hole is None:
                    first_hole = owner
                if fill != -1 and val is None:
                    val = ('fill', fill)
            if isinstance(val, tuple) and val[0] == 'fill':
                val = (val[1], first_hole)
            if val is None or val[0] == -1:
                continue
            if val[1] is not None:
                grid[(x, y)] = val
    return grid


def comps(cells):
    cells = set(cells)
    out = []
    while cells:
        st = [cells.pop()]
        c = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def bbox(c):
    xs = [p[0] for p in c]
    ys = [p[1] for p in c]
    return min(xs), min(ys), max(xs), max(ys)


def mkobj(cells, grid, typ, layer, tags):
    x0, y0, x1, y1 = bbox(cells)
    px = [[grid[(x, y)][0] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'type': typ, 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'layer': layer, 'tags': tags, 'pixels': px}


def cluster(cs):
    cs = [set(c) for c in cs]
    merged = True
    while merged:
        merged = False
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                ax0, ay0, ax1, ay1 = bbox(cs[a])
                bx0, by0, bx1, by1 = bbox(cs[b])
                gx = max(bx0 - ax1, ax0 - bx1) - 1
                gy = max(by0 - ay1, ay0 - by1) - 1
                if gx <= 3 and gy <= 3:
                    cs[a] |= cs.pop(b)
                    merged = True
                    break
            if merged:
                break
    return cs


def extract(m, grid, counter):
    out = [counter] if counter else []
    wall_cells = [p for p, v in grid.items() if v[1][0] == 'wall']
    piece_cells = [p for p, v in grid.items() if v[1][0] == 'piece']
    walls = []
    for c in comps(wall_cells):
        x0, y0, x1, y1 = bbox(c)
        o = mkobj(c, grid, 'wall', 1, ['axis', 'horizontal' if y1 - y0 > x1 - x0 else 'vertical'])
        walls.append(o)
    pcs = [mkobj(c, grid, 'player', 4, ['movable', 'black']) for c in comps(piece_cells)]
    z_piece = [p for p in piece_cells if grid[p][0] == 0]
    z_wall = [p for p in wall_cells if grid[p][0] == 0]
    keys = sorted([((o['x'], o['y']), 0, o) for o in walls] + [(p, 1, None) for p in z_piece], key=lambda t: (t[0], t[1]))
    for k, (_, t, o) in enumerate(keys):
        if o is not None:
            o['name'] = 'wall_%s_%d' % ('h' if o['tags'][1] == 'horizontal' else 'v', k)
    keys = sorted([((o['x'], o['y']), 0, o) for o in pcs] + [(p, 1, None) for p in z_wall], key=lambda t: (t[0], t[1]))
    for k, (_, t, o) in enumerate(keys):
        if o is not None:
            o['name'] = 'piece_%d' % k
    vis, inv = reflections(m)
    refl = [mkobj(c, grid, 'reflection', 3, ['mirror', 'gray']) for c in comps([p for p, v in grid.items() if v[1][0] == 'refl'])]
    ghost = [bbox({(3 * i, 3 * j) for i, j in c}) for c in comps(inv)]
    keys = sorted([((o['x'], o['y']), o) for o in refl] + [((g[0], g[1]), None) for g in ghost], key=lambda t: t[0])
    for k, (_, o) in enumerate(keys):
        if o is not None:
            o['name'] = 'reflection_%d' % k
    tg = sorted((mkobj(c, grid, 'target', 2, ['goal', 'yellow']) for c in cluster(comps([p for p, v in grid.items() if v[1][0] == 'target']))), key=lambda o: (o['x'], o['y']))
    for k, o in enumerate(tg):
        o['name'] = 'target_%d' % k
    return out + walls + pcs + refl + tg


def transition_function(state, action):
    m = parse(state)
    if _mem['out'] is not None and canon(state) == _mem['out'] and _mem['targets'] is not None:
        m['targets'] |= _mem['targets']
    counter = next((o for o in state if o['type'] == 'counter'), None)
    m = step(m, action)
    out = extract(m, render(m), counter)
    _mem['out'] = canon(out)
    _mem['targets'] = set(m['targets'])
    return out
