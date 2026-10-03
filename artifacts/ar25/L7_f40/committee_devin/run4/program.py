# Mechanics: two wall axes (h row band, v column band) and 2 holed pieces on a 3x3-block grid (x=row, pixels x-major).
# ACTION5 cycles selection h -> v -> piece0 -> piece1 -> h; A1/A2 move h axis or piece by -/+1 block row, A3/A4 move v axis or piece by columns.
# Reflections: v-mirror visible gray (ring 4, hole); h/double mirrors invisible rings that occlude but still count in names; none on piece blocks.
# Frame composited (wall<target(solid)<reflection<piece, holes show lower solid else own fill) and re-extracted: 4-conn ranks, dot-only skipped.
# Hypothesis 'reflection A1 gone vs reshaped' = which object is selected (0-holes) + re-ranking; unconfirmed: piece blocking, piece order after merges.
import json

N = 63
NB = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_mem = {'last': None, 'model': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def frame(state):
    f = {}
    for o in state:
        if 'pixels' not in o:
            continue
        for i, row in enumerate(o['pixels']):
            for j, v in enumerate(row):
                if v != -1:
                    f[(o['x'] + i, o['y'] + j)] = (o['type'], v)
    return f


def best_band(f, axis):
    cnt = [0] * NB
    for (x, y), (t, v) in f.items():
        if t == 'wall':
            cnt[(x if axis == 0 else y) // 3] += 1
    m = max(cnt)
    return cnt.index(m) if m else None


def block_sets(f, typ):
    return {(x // 3, y // 3) for (x, y), (t, v) in f.items() if t == typ}


def bcomps(blocks):
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
    return sorted(out, key=lambda c: (min(c), ))


def parse(state, prev):
    f = frame(state)
    ah = best_band(f, 0)
    av = best_band(f, 1)
    if prev is not None:
        ah = prev['ah'] if ah is None else ah
        av = prev['av'] if av is None else av
    pblocks = {(x // 3, y // 3) for (x, y), (t, v) in f.items()
               if t == 'player' and (x % 3, y % 3) != (1, 1)}
    if prev is not None and set().union(*prev['pieces']) == pblocks:
        pieces = [set(p) for p in prev['pieces']]
    else:
        pieces = bcomps(pblocks)
        pieces.sort(key=lambda p: (min(b[0] for b in p) * 3, min(b[1] for b in p)))
    zeros = [c for c, (t, v) in f.items() if v == 0]
    zp = [c for c in zeros if (c[0] // 3, c[1] // 3) in pblocks]
    zh = [c for c in zeros if c[0] == 3 * ah + 1 and c[1] // 3 != av and c not in zp]
    zv = [c for c in zeros if c[1] == 3 * av + 1 and c[0] // 3 != ah and c not in zp]
    if zh:
        sel = 'h'
    elif zv:
        sel = 'v'
    elif zp:
        b = (zp[0][0] // 3, zp[0][1] // 3)
        sel = [i for i, p in enumerate(pieces) if b in p][0]
    elif prev is not None:
        sel = prev['sel']
    else:
        sel = 'v' if (3 * ah + 1, 3 * av + 1) in zeros else 'h'
    tg = block_sets(f, 'target')
    if prev is not None:
        tg |= prev['targets']
    return {'ah': ah, 'av': av, 'sel': sel, 'pieces': pieces, 'targets': tg}


def step(m, action):
    m = {'ah': m['ah'], 'av': m['av'], 'sel': m['sel'],
         'pieces': [set(p) for p in m['pieces']], 'targets': set(m['targets'])}
    if action == 5:
        order = ['h', 'v'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)]
        return m
    if action not in DIRS:
        return m
    dx, dy = DIRS[action]
    if m['sel'] == 'h':
        if dx and 0 <= m['ah'] + dx < NB:
            m['ah'] += dx
    elif m['sel'] == 'v':
        if dy and 0 <= m['av'] + dy < NB:
            m['av'] += dy
    else:
        p = m['pieces'][m['sel']]
        moved = {(a + dx, b + dy) for a, b in p}
        others = set().union(set(), *[q for i, q in enumerate(m['pieces']) if i != m['sel']])
        if all(0 <= a < NB and 0 <= b < NB for a, b in moved) and not (moved & others):
            m['pieces'][m['sel']] = moved
    return m


def block_sprite(blocks, ring, fill, solid=False):
    cells = {}
    for a, b in blocks:
        for i in range(3):
            for j in range(3):
                c = (3 * a + i, 3 * b + j)
                cells[c] = ('s' if solid else 'h', fill) if (i, j) == (1, 1) else ('s', ring)
    return cells


def render(m):
    ah, av = m['ah'], m['av']
    sprites = []
    sprites.append(('wall', block_sprite({(ah, j) for j in range(NB)}, 10, 0 if m['sel'] == 'h' else -1)))
    sprites.append(('wall', block_sprite({(i, av) for i in range(NB)}, 10, 0 if m['sel'] == 'v' else -1)))
    sprites.append(('target', block_sprite(m['targets'], 11, 11, True)))
    occ = set().union(set(), *m['pieces'])
    vis, inv = set(), set()
    for a, b in occ:
        vis.add((a, 2 * av - b))
        inv.add((2 * ah - a, b))
        inv.add((2 * ah - a, 2 * av - b))
    ok = lambda s: {(a, b) for a, b in s if 0 <= a < NB and 0 <= b < NB and (a, b) not in occ}
    sprites.append(('reflection', block_sprite(ok(inv), 'X', 'Y')))
    sprites.append(('reflection', block_sprite(ok(vis), 4, 4)))
    for i, p in enumerate(m['pieces']):
        sprites.append(('player', block_sprite(p, 5, 0 if m['sel'] == i else -1)))
    out = {}
    for x in range(N):
        for y in range(N):
            hole = None
            for typ, cells in reversed(sprites):
                k = cells.get((x, y))
                if k is None:
                    continue
                if k[0] == 's':
                    out[(x, y)] = (typ, k[1])
                    break
                if hole is None and k[1] != -1:
                    hole = (typ, k[1])
            else:
                if hole is not None:
                    out[(x, y)] = hole
    return out


def comps(cells):
    cells, res = set(cells), []
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
        res.append(comp)
    return res


def bbox(cs):
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]
    return min(xs), min(ys), max(xs), max(ys)


def make(name, typ, layer, tags, cells, fr):
    x0, y0, x1, y1 = bbox(cells)
    px = [[-1] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    for c in cells:
        v = fr[c][1]
        px[c[0] - x0][c[1] - y0] = -1 if v in ('X', 'Y') else v
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
            'layer': layer, 'tags': tags, 'pixels': px}


def merge_targets(cs, gap=3):
    cs = [set(c) for c in cs]
    changed = True
    while changed:
        changed = False
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = bbox(cs[i]), bbox(cs[j])
                dx = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
                dy = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
                if max(dx, dy) <= gap:
                    cs[i] |= cs.pop(j)
                    changed = True
                    break
            if changed:
                break
    return cs


def extract(fr, counter):
    objs = [dict(counter)] if counter else []
    zeros = {c for c, (t, v) in fr.items() if v == 0}
    for typ, layer in (('wall', 1), ('player', 4)):
        own = {c for c, (t, v) in fr.items() if t == typ}
        cs = sorted(comps(own | zeros), key=lambda c: bbox(c)[:2])
        for k, c in enumerate(cs):
            if not any(fr[e][1] != 0 for e in c & own):
                continue
            o = make('', typ, layer, [], c, fr)
            if typ == 'wall':
                hz = o['h'] > o['w']
                o['name'] = 'wall_%s_%d' % ('h' if hz else 'v', k)
                o['tags'] = ['axis', 'horizontal' if hz else 'vertical']
            else:
                o['name'] = 'piece_%d' % k
                o['tags'] = ['movable', 'black']
            objs.append(o)
    own = {c for c, (t, v) in fr.items() if t == 'reflection'}
    cs = sorted(comps(own), key=lambda c: bbox(c)[:2])
    for k, c in enumerate(cs):
        visc = {e for e in c if fr[e][1] not in ('X', 'Y')}
        if visc:
            objs.append(make('reflection_%d' % k, 'reflection', 3, ['mirror', 'gray'], visc, fr))
    own = {c for c, (t, v) in fr.items() if t == 'target'}
    cs = sorted(merge_targets(comps(own)), key=lambda c: bbox(c)[:2])
    for k, c in enumerate(cs):
        objs.append(make('target_%d' % k, 'target', 2, ['goal', 'yellow'], c, fr))
    return objs


def transition_function(state, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    prev = _mem['model'] if _mem['last'] is not None and canon(state) == _mem['last'] else None
    m = parse(state, prev)
    m2 = step(m, aid)
    counter = next((o for o in state if o['type'] == 'counter'), None)
    out = extract(render(m2), counter)
    _mem['last'] = canon(out)
    _mem['model'] = m2
    return out
