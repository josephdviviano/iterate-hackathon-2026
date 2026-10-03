# Mechanics: single horizontal mirror axis (wall row, 3x3 blocks) + holed pieces + gray reflections + yellow targets.
# A1/A2 move the selection (axis or piece) up/down a block, A3/A4 move a piece left/right (bounds, no piece overlap);
# A5 cycles axis -> pieces (largest first) -> axis; A7 undoes the last position change (selection kept, free).
# Budget counter: 64 free actions, then each action shrinks the HUD bar by 1 cell; once it shrinks, the spent
# segment ranks as one extra component before the pieces (piece_i index +1). Hypothesis: spent HUD cells are drawn top-left.
import json

N = 21  # board blocks per side (63 cells)


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    counter = next((dict(o) for o in state if o['type'] == 'counter'), None)
    ax = walls[0]['x'] // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    vis = {}
    for o in state:
        for x, y, v in cells(o):
            vis[(x, y)] = v
    targets = []
    for o in sorted(state, key=lambda o: o['name']):
        if o['type'] == 'target':
            targets.append(frozenset((x // 3, y // 3) for x, y, _ in cells(o)))
    pieces, sel = [], None
    for o in state:
        if o['type'] != 'player':
            continue
        B = {(x // 3, y // 3) for x, y, v in cells(o) if v == 5 and x % 3 == 0 and y % 3 == 0}
        cen = {b: vis.get((3 * b[0] + 1, 3 * b[1] + 1), -1) for b in B}
        Z = {b for b in B if cen[b] == 0}
        if Z and Z != B and not axis_sel:
            amb = {b for b in B if cen[b] not in (0, -1)}
            grow = True
            while grow:
                grow = False
                for b in list(amb - Z):
                    if any((b[0] + dx, b[1] + dy) in Z for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        Z.add(b)
                        grow = True
            pieces.append(frozenset(Z))
            sel = len(pieces) - 1
            if B - Z:
                pieces.append(frozenset(B - Z))
        else:
            pieces.append(frozenset(B))
            if Z and not axis_sel:
                sel = len(pieces) - 1
    if axis_sel:
        sel = 'axis'
    elif sel is None:
        for i, P in enumerate(pieces):
            if all(vis.get((3 * a + 1, 3 * b + 1), -1) not in (-1, 0) for a, b in P):
                sel = i
                break
    n = 0
    if counter and counter['h'] < 64:
        n = 128 - counter['h']
    return {'ax': ax, 'sel': sel, 'pieces': pieces, 'targets': targets, 'n': n, 'hist': [],
            'counter': counter}


def cycle_order(m):
    idx = sorted(range(len(m['pieces'])), key=lambda i: (-len(m['pieces'][i]), min(m['pieces'][i])))
    return ['axis'] + idx


def step(m, action):
    m = dict(m)
    m['pieces'] = list(m['pieces'])
    m['hist'] = list(m['hist'])
    aid = action['action_id'] if isinstance(action, dict) else action
    before = (m['ax'], tuple(m['pieces']))
    if aid in (1, 2, 3, 4):
        dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[aid]
        if m['sel'] == 'axis':
            if dy == 0 and 0 <= m['ax'] + dx < N:
                m['ax'] += dx
        elif m['sel'] is not None:
            i = m['sel']
            new = frozenset((a + dx, b + dy) for a, b in m['pieces'][i])
            others = set().union(*[P for j, P in enumerate(m['pieces']) if j != i])
            if all(0 <= a < N and 0 <= b < N for a, b in new) and not (new & others):
                m['pieces'][i] = new
    elif aid == 5:
        order = cycle_order(m)
        m['sel'] = 'axis' if m['sel'] not in order else order[(order.index(m['sel']) + 1) % len(order)]
    elif aid == 7:
        if m['hist']:
            m['ax'], ps = m['hist'].pop()
            m['pieces'] = list(ps)
    if aid != 7:
        m['n'] += 1
        if (m['ax'], tuple(m['pieces'])) != before:
            m['hist'].append(before)
    return m


def render(m):
    """Per-block sprite stacks top-down -> cell grid of (colour, kind, id)."""
    ax, sel = m['ax'], m['sel']
    stacks = {}

    def push(b, spr):
        stacks.setdefault(b, []).append(spr)
    pblocks = set().union(*m['pieces']) if m['pieces'] else set()
    for i, P in enumerate(m['pieces']):
        for b in P:
            push(b, ('player', i, 5, None, 0 if sel == i else None))
    refl = set()
    for a, c in pblocks:
        r = 2 * ax - a
        if a != ax and 0 <= r < N and (r, c) not in pblocks:
            refl.add((r, c))
    for b in refl:
        push(b, ('reflection', 0, 4, None, 4))
    for k, T in enumerate(m['targets']):
        for b in T:
            push(b, ('target', k, 11, 11, None))
    for c in range(N):
        push((ax, c), ('wall', 0, 10, 0 if sel == 'axis' else None, None))
    grid = {}
    for (a, c), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    grid[(3 * a + i, 3 * c + j)] = (top[2], top[0], top[1])
        solid = next((s for s in st if s[3] is not None), None)
        if solid:
            grid[(3 * a + 1, 3 * c + 1)] = (solid[3], solid[0], solid[1])
        elif top[4] is not None:
            grid[(3 * a + 1, 3 * c + 1)] = (top[4], top[0], top[1])
    return grid


def comps(cellset):
    seen, out = set(), []
    for c in sorted(cellset):
        if c in seen:
            continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cellset and n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(comp)
    out.sort(key=lambda cp: (min(p[0] for p in cp), min(p[1] for p in cp)))
    return out


def make_obj(name, typ, layer, tags, comp, colour):
    x0 = min(p[0] for p in comp)
    y0 = min(p[1] for p in comp)
    w = max(p[0] for p in comp) - x0 + 1
    h = max(p[1] for p in comp) - y0 + 1
    px = [[-1] * h for _ in range(w)]
    for p in comp:
        px[p[0] - x0][p[1] - y0] = colour[p]
    return {'name': name, 'type': typ, 'layer': layer, 'tags': tags, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'pixels': px}


def extract(m, grid):
    out = []
    col = {p: v[0] for p, v in grid.items()}
    zeros = {p for p, v in col.items() if v == 0}
    spent = 1 if m['counter'] and m['counter']['h'] < 64 else 0
    for typ, colour, layer, tags, prefix, offset in (
            ('wall', 10, 1, ['axis', 'horizontal'], 'wall_h_', 0),
            ('player', 5, 4, ['movable', 'black'], 'piece_', spent)):
        own = {p for p, v in col.items() if v == colour}
        for k, cp in enumerate(comps(own | zeros)):
            if any(p in own for p in cp):
                out.append(make_obj(prefix + str(k + offset), typ, layer, tags, cp, col))
    for k, cp in enumerate(comps({p for p, v in col.items() if v == 4})):
        out.append(make_obj('reflection_%d' % k, 'reflection', 3, ['mirror', 'gray'], cp, col))
    tcomps = []
    for t in range(len(m['targets'])):
        cs = [p for p, v in grid.items() if v[1] == 'target' and v[2] == t]
        if cs:
            tcomps.append(cs)
    tcomps.sort(key=lambda cp: (min(p[0] for p in cp), min(p[1] for p in cp)))
    for k, cp in enumerate(tcomps):
        out.append(make_obj('target_%d' % k, 'target', 2, ['goal', 'yellow'], cp, col))
    if m['counter']:
        out.append(m['counter'])
    return out


_last = {'canon': None, 'model': None}


def transition_function(state, action):
    if _last['canon'] is not None and canon(state) == _last['canon']:
        m = _last['model']
    else:
        m = parse(state)
    m = step(m, action)
    if m['counter']:
        c = dict(m['counter'])
        c['h'] = min(64, 128 - m['n'])
        c['y'] = 64 - c['h']
        m['counter'] = c
    out = extract(m, render(m))
    _last['canon'] = canon(out)
    _last['model'] = m
    return out
