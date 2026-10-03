# Mechanics: two mirror axes (h-axis = 3-row band, v-axis = 3-col band, 3x3 holed cells) + 3x3-block pieces on a 63x63 board.
# ACTION5 cycles selection h-axis -> v-axis -> pieces (largest first) -> h-axis; selected holes render 0. A1/A2 move h-axis or piece x-/+3, A3/A4 move v-axis or piece y-/+3.
# Pieces are reflected across the v-axis (visible gray 4), h-axis and both (invisible occluders); render layers wall1<target2<reflection3<piece4, then re-extract 4-conn components.
# Names = type rank by bbox (x,y), counting 0 pixels of the other type (wall<->player); merged wall tag = horizontal iff h > w. Pieces block on bounds/other pieces/axis bands.
# Unconfirmed: piece selection order (only one piece ever selected), blocking rules, invisible-reflection rendering (all off-board in the observations).
import json

N = 63
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
_memo = {'canon': None, 'model': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def comps(cells):
    left, res = set(cells), []
    while left:
        s = left.pop(); st, c = [s], {s}
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in left:
                    left.discard(n); c.add(n); st.append(n)
        res.append(c)
    return res


def blk(c):
    return (c[0] // 3 * 3, c[1] // 3 * 3)


def parse(state):
    walls, players, owner, targets, extra = {}, {}, {}, [], []
    for o in state:
        t = o['type']
        cs = cells_of(o)
        if t == 'wall':
            walls.update(cs)
        elif t == 'player':
            players.update(cs)
        elif t == 'target':
            tb = {}
            for c, v in cs.items():
                tb.setdefault(blk(c), v)
            targets.append({'name': o['name'], 'tags': o['tags'], 'blocks': tb})
        elif t not in ('reflection',):
            extra.append(o)
        for c, v in cs.items():
            owner[c] = (t, v)

    def band(axis):
        cnt = {}
        for c in walls:
            cnt[c[axis] // 3] = cnt.get(c[axis] // 3, 0) + 1
        return 3 * max(cnt, key=lambda k: cnt[k]) + 1 if cnt else None
    hc, vc = band(0), band(1)
    sel = None
    for c, v in walls.items():
        if v == 0:
            if c[0] == hc and c[1] != vc:
                sel = 'h'
            elif c[1] == vc and c[0] != hc:
                sel = 'v'
    if sel is None and any(v == 0 for c, v in walls.items()):
        sel = 'h'
    pieces = []
    for comp in comps(players):
        bl = {blk(c) for c in comp}
        stat = {}
        for b in bl:
            ctr = (b[0] + 1, b[1] + 1)
            if players.get(ctr) == 0:
                stat[b] = 'sel'
            elif ctr in owner:
                stat[b] = 'amb'
            else:
                stat[b] = 'un'
        s = {b for b in bl if stat[b] == 'sel'}
        if not s:
            pieces.append({'blocks': bl, 'sel': False})
            continue
        grow = True
        while grow:
            grow = False
            for b in bl:
                if b not in s and stat[b] == 'amb' and any(
                        (b[0] + dx, b[1] + dy) in s for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3))):
                    s.add(b); grow = True
        pieces.append({'blocks': s, 'sel': True})
        rest = bl - s
        for rc in comps({(a // 3, b // 3) for a, b in rest}):
            pieces.append({'blocks': {(a * 3, b * 3) for a, b in rc}, 'sel': False})
    pieces.sort(key=lambda p: (-len(p['blocks']), min(p['blocks'])))
    for i, p in enumerate(pieces):
        if p.pop('sel'):
            sel = i
    return {'hc': hc, 'vc': vc, 'sel': sel, 'pieces': pieces, 'targets': targets, 'extra': extra}


def piece_cells(bl):
    out = set()
    for a, b in bl:
        out |= {(a + i, b + j) for i in range(3) for j in range(3)}
    return out


def step(m, action):
    hc, vc, sel, pieces = m['hc'], m['vc'], m['sel'], m['pieces']
    order = (['h'] if hc is not None else []) + (['v'] if vc is not None else []) + list(range(len(pieces)))
    if action == 5:
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else (order[0] if order else None)
        return
    d = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(action)
    if d is None or sel is None:
        return
    occupied = [piece_cells(p['blocks']) for p in pieces]
    allc = set().union(*occupied) if occupied else set()
    if sel == 'h' and d[0]:
        n = hc + d[0]
        if 1 <= n <= N - 2 and not any(abs(c[0] - n) <= 1 for c in allc):
            m['hc'] = n
    elif sel == 'v' and d[1]:
        n = vc + d[1]
        if 1 <= n <= N - 2 and not any(abs(c[1] - n) <= 1 for c in allc):
            m['vc'] = n
    elif isinstance(sel, int):
        nb = {(a + d[0], b + d[1]) for a, b in pieces[sel]['blocks']}
        nc = piece_cells(nb)
        others = set().union(set(), *[occupied[i] for i in range(len(pieces)) if i != sel])
        ok = all(0 <= a < N and 0 <= b < N for a, b in nc) and not (nc & others)
        if hc is not None and any(abs(a - hc) <= 1 for a, b in nc):
            ok = False
        if vc is not None and any(abs(b - vc) <= 1 for a, b in nc):
            ok = False
        if ok:
            pieces[sel]['blocks'] = nb


def block_sprite(bl, colour, fill):
    spr = {}
    for a, b in bl:
        for i in range(3):
            for j in range(3):
                c = (a + i, b + j)
                if 0 <= c[0] < N and 0 <= c[1] < N:
                    spr[c] = ('h', fill) if (i, j) == (1, 1) else ('s', colour)
    return spr


def render(m):
    hc, vc, sel = m['hc'], m['vc'], m['sel']
    sprites = []  # (layer, kind, id, {cell: ('s', colour) | ('h', fill)})
    for ax, c0, s in (('h', hc, 0), ('v', vc, 1)):
        if c0 is None:
            continue
        spr = {}
        for k in range(N):
            for off in (-1, 0, 1):
                cell = (c0 + off, k) if s == 0 else (k, c0 + off)
                spr[cell] = ('h', 0 if sel == ax else -1) if off == 0 and k % 3 == 1 else ('s', 10)
        sprites.append((1, 'wall', ax, spr))
    for ti, t in enumerate(m['targets']):
        spr = {}
        for (a, b), v in t['blocks'].items():
            for i in range(3):
                for j in range(3):
                    spr[(a + i, b + j)] = ('s', v)
        sprites.append((2, 'target', ti, spr))
    occ = set()
    for p in m['pieces']:
        occ |= p['blocks']
    vis, invis = set(), set()
    for p in m['pieces']:
        for a, b in p['blocks']:
            rv = (a, 2 * vc - b - 2) if vc is not None else None
            rh = (2 * hc - a - 2, b) if hc is not None else None
            rb = (2 * hc - a - 2, 2 * vc - b - 2) if hc is not None and vc is not None else None
            if rv and rv not in occ:
                vis.add(rv)
            for r in (rh, rb):
                if r and r not in occ:
                    invis.add(r)
    invis -= vis
    sprites.append((3, 'reflection', 'vis', block_sprite(vis, 4, 4)))
    sprites.append((3, 'reflection', 'invis', block_sprite(invis, None, 4)))
    for i, p in enumerate(m['pieces']):
        sprites.append((4, 'player', i, block_sprite(p['blocks'], 5, 0 if sel == i else -1)))
    sprites.sort(key=lambda s: -s[0])
    cellset = set()
    for s in sprites:
        cellset |= set(s[3])
    shown = {}
    for c in cellset:
        stack = [(s[1], s[2], s[3][c]) for s in sprites if c in s[3]]
        top = next((x for x in stack if x[2][0] == 's'), None)
        if top is None:
            top = next((x for x in stack if x[2][1] >= 0), None)
        if top is not None:
            shown[c] = (top[0], top[1], top[2][1])
    return shown


def mkobj(name, t, cells, vals, tags):
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[-1] * h for _ in range(w)]
    for c in cells:
        px[c[0] - x0][c[1] - y0] = vals[c]
    return {'name': name, 'type': t, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[t], 'tags': tags, 'pixels': px}


def extract(m, shown):
    out = [dict(o) for o in m['extra']]
    by = {}
    for c, (t, i, v) in shown.items():
        by.setdefault(t, {})[c] = (i, v)
    vals = {c: v for c, (t, i, v) in shown.items()}
    walls = [mkobj('', 'wall', cs, vals, ['axis', 'horizontal']) for cs in comps(by.get('wall', {}))]
    for o in walls:
        o['tags'] = ['axis', 'horizontal' if o['h'] > o['w'] else 'vertical']
    players = [mkobj('', 'player', cs, vals, ['movable', 'black']) for cs in comps(by.get('player', {}))]
    refl, rcount = [], []
    for cs in comps(by.get('reflection', {})):
        visc = {c for c in cs if vals[c] is not None}
        if visc:
            refl.append(mkobj('', 'reflection', visc, vals, ['mirror', 'gray']))
        else:
            rcount.append((min(cs)))
    targets = []
    for ti, t in enumerate(m['targets']):
        cs = {c for c, (i, v) in by.get('target', {}).items() if i == ti}
        if cs:
            o = mkobj('', 'target', cs, vals, t['tags'])
            targets.append(o)

    def zeros(objs):
        return [(o['x'] + i, o['y'] + j) for o in objs for i, r in enumerate(o['pixels'])
                for j, v in enumerate(r) if v == 0]

    def name(objs, prefix, extra_keys):
        objs.sort(key=lambda o: (o['x'], o['y']))
        for o in objs:
            k = (o['x'], o['y'])
            idx = sum(1 for p in objs if (p['x'], p['y']) < k) + sum(1 for e in extra_keys if e < k)
            o['name'] = prefix(o) + '_' + str(idx)
    name(walls, lambda o: 'wall_h' if 'horizontal' in o['tags'] else 'wall_v', zeros(players))
    name(players, lambda o: 'piece', zeros(walls))
    name(refl, lambda o: 'reflection', rcount)
    name(targets, lambda o: 'target', [])
    return out + walls + players + refl + targets


def transition_function(state, action):
    import copy
    key = canon(state)
    if _memo['canon'] == key and _memo['model'] is not None:
        m = copy.deepcopy(_memo['model'])
    else:
        m = parse(state)
    a = action['action_id'] if isinstance(action, dict) else action
    step(m, a)
    out = extract(m, render(m))
    _memo['canon'], _memo['model'] = canon(out), copy.deepcopy(m)
    return out
