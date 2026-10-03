# Mechanics: one horizontal mirror axis (wall row band) + holed 3x3-block pieces on a 21x21 block grid (x=row).
# A1/A2/A3/A4 move the selection by one block (axis: rows only; pieces: blocked by bounds/other pieces);
# A5 cycles axis -> pieces (largest first) -> axis; A7 undoes the last successful move (selection kept, free).
# Pieces reflect across the axis as gray blocks; frame is layered (wall<target<reflection<piece) and re-extracted.
# Budget bar: every non-undo action costs 1, bar h = min(64, 128-n); hypothesis: spent cells rank first among players.
import json

N = 21
_mem = {}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _cells(state):
    g = {}
    for k, o in enumerate(state):
        for i, row in enumerate(o.get('pixels') or []):
            for j, v in enumerate(row):
                if v != -1:
                    g[(o['x'] + i, o['y'] + j)] = (v, o['type'], k)
    return g


def _bcomps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in blocks:
                    blocks.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def parse(state):
    g = _cells(state)
    walls = [o for o in state if o['type'] == 'wall']
    A = min(o['x'] for o in walls) // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for row in o['pixels'] for v in row)
    pblocks = [(r, c) for r in range(N) for c in range(N)
               if g.get((3 * r, 3 * c), (None, None))[1] == 'player' and g[(3 * r, 3 * c)][0] == 5]
    pieces, sel = [], 'axis' if axis_sel else None
    for comp in _bcomps(pblocks):
        kind = {}
        for (r, c) in comp:
            cell = g.get((3 * r + 1, 3 * c + 1))
            if cell is None:
                kind[(r, c)] = 'empty'
            elif cell[0] == 0 and cell[1] == 'player' and not (axis_sel and r == A):
                kind[(r, c)] = 'sel'
            else:
                kind[(r, c)] = 'amb'
        seeds = [b for b in comp if kind[b] == 'sel']
        if seeds:
            grown, st = set(seeds), list(seeds)
            while st:
                r, c = st.pop()
                for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if nb in comp and nb not in grown and kind[nb] == 'amb':
                        grown.add(nb)
                        st.append(nb)
            sel = len(pieces)
            pieces.append(frozenset(grown))
            for rest in _bcomps(comp - grown):
                pieces.append(frozenset(rest))
        else:
            if sel is None and all(kind[b] == 'amb' for b in comp):
                sel = len(pieces)
            pieces.append(frozenset(comp))
    if sel is None:
        sel = 'axis'
    targets = []
    for k, o in enumerate(state):
        if o['type'] == 'target':
            bl = {(p[0] // 3, p[1] // 3) for p, v in g.items() if v[2] == k}
            targets.append(frozenset(bl))
    counter = [o for o in state if o['type'] == 'counter']
    spent = counter[0]['y'] if counter else 0
    n = 64 + spent if spent > 0 else 0
    return {'A': A, 'sel': sel, 'pieces': pieces, 'targets': targets, 'n': n, 'undo': []}


def cycle(m):
    order = sorted(range(len(m['pieces'])), key=lambda i: (-len(m['pieces'][i]), min(m['pieces'][i])))
    seq = ['axis'] + order
    if m['sel'] not in seq:
        return 'axis'
    return seq[(seq.index(m['sel']) + 1) % len(seq)]


def step(m, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    if aid == 7:
        if m['undo']:
            m['A'], m['pieces'] = m['undo'].pop()
        return m
    m['n'] += 1
    if aid == 5:
        m['sel'] = cycle(m)
    elif aid in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[aid]
        snap = (m['A'], list(m['pieces']))
        if m['sel'] == 'axis':
            if dr and 0 <= m['A'] + dr < N:
                m['A'] += dr
                m['undo'].append(snap)
        else:
            i = m['sel']
            nb = frozenset((r + dr, c + dc) for r, c in m['pieces'][i])
            others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i])
            if all(0 <= r < N and 0 <= c < N for r, c in nb) and not (nb & others):
                m['pieces'] = list(m['pieces'])
                m['pieces'][i] = nb
                m['undo'].append(snap)
    return m


def render(m):
    A, sel = m['A'], m['sel']
    pmap = {b: i for i, p in enumerate(m['pieces']) for b in p}
    refl = {(2 * A - r, c) for p in m['pieces'] for r, c in p if r != A and 0 <= 2 * A - r < N} - set(pmap)
    tmap = {b: k for k, t in enumerate(m['targets']) for b in t}
    g = {}
    for r in range(N):
        for c in range(N):
            stack = []
            if (r, c) in pmap:
                stack.append((5, 0 if sel == pmap[(r, c)] else -1, False, 'player', pmap[(r, c)]))
            if (r, c) in refl:
                stack.append((4, 4, False, 'reflection', 0))
            if (r, c) in tmap:
                stack.append((11, 11, True, 'target', tmap[(r, c)]))
            if r == A:
                stack.append((10, 0 if sel == 'axis' else -1, sel == 'axis', 'wall', 0))
            if not stack:
                continue
            top = stack[0]
            for i in range(3):
                for j in range(3):
                    if (i, j) != (1, 1):
                        g[(3 * r + i, 3 * c + j)] = (top[0], top[3], top[4])
            solid = next((s for s in stack[1:] if s[2]), None) if not top[2] else top
            if solid is not None:
                g[(3 * r + 1, 3 * c + 1)] = (solid[1], solid[3], solid[4])
            elif top[1] != -1:
                g[(3 * r + 1, 3 * c + 1)] = (top[1], top[3], top[4])
    return g


def _pcomps(cellset):
    cellset, out = set(cellset), []
    while cellset:
        st = [cellset.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cellset:
                    cellset.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def _obj(name, typ, layer, tags, comp, g):
    r0, c0 = min(p[0] for p in comp), min(p[1] for p in comp)
    r1, c1 = max(p[0] for p in comp), max(p[1] for p in comp)
    px = [[g[(r, c)][0] if (r, c) in comp else -1 for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return {'name': name, 'type': typ, 'x': r0, 'y': c0, 'w': r1 - r0 + 1, 'h': c1 - c0 + 1,
            'layer': layer, 'tags': tags, 'pixels': px}


def _ranked(comps, own, g, prefix, typ, layer, tags, extra=()):
    keyed = [((min(p[0] for p in cp), min(p[1] for p in cp)), cp) for cp in comps] + list(extra)
    keyed.sort(key=lambda kc: kc[0])
    out = []
    for i, (_, cp) in enumerate(keyed):
        if cp and any(g[p][0] == own for p in cp):
            out.append(_obj(prefix + str(i), typ, layer, tags, cp, g))
    return out


def extract(m, g):
    out = []
    spent = max(0, m['n'] - 64)
    zero = {p for p, v in g.items() if v[0] == 0}
    out += _ranked(_pcomps({p for p, v in g.items() if v[0] == 10} | zero), 10, g,
                   'wall_h_', 'wall', 1, ['axis', 'horizontal'])
    budget = [((0, 63), set())] if spent else []
    out += _ranked(_pcomps({p for p, v in g.items() if v[0] == 5} | zero), 5, g,
                   'piece_', 'player', 4, ['movable', 'black'], budget)
    out += _ranked(_pcomps({p for p, v in g.items() if v[0] == 4}), 4, g,
                   'reflection_', 'reflection', 3, ['mirror', 'gray'])
    tcomps = []
    for k in range(len(m['targets'])):
        cp = {p for p, v in g.items() if v[1] == 'target' and v[2] == k}
        if cp:
            tcomps.append(cp)
    out += _ranked(tcomps, 11, g, 'target_', 'target', 2, ['goal', 'yellow'])
    out.append({'name': 'counter', 'type': 'counter', 'x': 63, 'y': spent, 'w': 1, 'h': 64 - spent,
                'layer': 5, 'tags': ['hud', 'budget']})
    return out


def transition_function(state, action):
    if _mem.get('last') == _canon(state):
        m = _mem['model']
        m = {**m, 'pieces': list(m['pieces']), 'undo': list(m['undo'])}
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m, render(m))
    _mem['last'], _mem['model'] = _canon(out), m
    return out
