# Mechanics: 3x3-block mirror game. World = horizontal axis row A, piece block sets, static target block sets,
# selection (axis or one piece), hidden budget B. A1/A2/A3/A4 move the selection by one block (axis: rows only),
# A5 cycles axis -> pieces by (size desc, min block) -> axis, A7 undoes the last move (free); actions 1-6 cost 1.
# Gray reflections mirror every piece block across A; layers wall<target<reflection<piece; frame re-extracted.
# Hypotheses: budget starts at 128 (bar shows min(B,64)); the spent bar segment ranks first among player comps.
import json

N = 21
COL = {'piece': 5, 'refl': 4, 'target': 11, 'wall': 10}
_mem = {'last': None, 'world': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def block_comps(blocks):
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
    grid = {}
    for o in state:
        for i, j, v in cells(o):
            grid[(i, j)] = v
    walls = [o for o in state if o['type'] == 'wall']
    A = walls[0]['x'] // 3 if walls else 10
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    pieces, sel = [], None
    for o in state:
        if o['type'] != 'player':
            continue
        blocks = {(i // 3, j // 3) for i, j, v in cells(o) if v == 5}
        cen = {b: grid.get((3 * b[0] + 1, 3 * b[1] + 1), -1) for b in blocks}
        zero = {b for b in blocks if cen[b] == 0 and not (axis_sel and b[0] == A)}
        if zero:
            part, st = set(zero), list(zero)
            while st:
                r, c = st.pop()
                for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if nb in blocks and nb not in part and cen[nb] != -1:
                        part.add(nb)
                        st.append(nb)
            sel = len(pieces)
            pieces.append(part)
            pieces.extend(block_comps(blocks - part))
        else:
            pieces.extend(block_comps(blocks))
    if axis_sel:
        sel = 'axis'
    elif sel is None:
        covered = [k for k, p in enumerate(pieces)
                   if all(grid.get((3 * r + 1, 3 * c + 1), -1) not in (-1, 0) for r, c in p)]
        sel = covered[0] if covered else None
    targets = []
    for o in sorted(state, key=lambda o: o['name']):
        if o['type'] == 'target':
            targets.append({(i // 3, j // 3) for i, j, v in cells(o) if v == 11})
    ctr = [o for o in state if o['type'] == 'counter']
    n = ctr[0]['h'] if ctr else 64
    return {'A': A, 'sel': sel, 'pieces': pieces, 'targets': targets,
            'B': n if n < 64 else 128, 'undo': [], 'ctr': ctr[0] if ctr else None}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda k: (-len(pieces[k]), min(pieces[k])))


def step(w, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    w = dict(w, pieces=[set(p) for p in w['pieces']], undo=list(w['undo']))
    if aid == 7:
        if w['undo']:
            w['A'], w['pieces'] = w['undo'].pop()
        return w
    w['B'] = max(0, w['B'] - 1)
    if aid in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[aid]
        snap = (w['A'], [set(p) for p in w['pieces']])
        if w['sel'] == 'axis':
            if dr and 0 <= w['A'] + dr < N:
                w['undo'].append(snap)
                w['A'] += dr
        elif w['sel'] is not None:
            moved = {(r + dr, c + dc) for r, c in w['pieces'][w['sel']]}
            if all(0 <= r < N and 0 <= c < N for r, c in moved):
                w['undo'].append(snap)
                w['pieces'][w['sel']] = moved
    elif aid == 5:
        cyc = ['axis'] + order(w['pieces'])
        w['sel'] = cyc[(cyc.index(w['sel']) + 1) % len(cyc)] if w['sel'] in cyc else 'axis'
    return w


def render(w):
    A, sel = w['A'], w['sel']
    pb = {}
    for k, p in enumerate(w['pieces']):
        for b in p:
            pb[b] = k
    rb = {(2 * A - r, c) for p in w['pieces'] for r, c in p}
    tb = {}
    for k, t in enumerate(w['targets']):
        for b in t:
            tb[b] = k
    g, owner = {}, {}
    for i in range(3 * N):
        for j in range(3 * N):
            b = (i // 3, j // 3)
            stack = []
            if b in pb:
                stack.append(('piece', 0 if pb[b] == sel else -1))
            if b in rb:
                stack.append(('refl', 4))
            if b in tb:
                stack.append(('target', None))
            if b[0] == A:
                stack.append(('wall', 0 if sel == 'axis' else None))
            if not stack:
                continue
            if i % 3 != 1 or j % 3 != 1:
                kind = stack[0][0]
                g[(i, j)] = COL[kind]
            else:
                kind, v = stack[0][0], stack[0][1] if stack[0][1] is not None else -1
                for k2, f in stack:
                    if k2 == 'target':
                        kind, v = 'target', 11
                        break
                    if k2 == 'wall' and f == 0:
                        kind, v = 'wall', 0
                        break
                if v < 0:
                    continue
                g[(i, j)] = v
            if kind == 'target':
                owner[(i, j)] = tb[b]
    return g, owner


def comps(g, colours):
    left = {p for p, v in g.items() if v in colours}
    out = []
    while left:
        st = [left.pop()]
        comp = set(st)
        while st:
            i, j = st.pop()
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if nb in left:
                    left.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return sorted(out, key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))


def make(name, typ, tags, layer, cellset, g):
    x0, y0 = min(p[0] for p in cellset), min(p[1] for p in cellset)
    x1, y1 = max(p[0] for p in cellset), max(p[1] for p in cellset)
    pix = [[g[(i, j)] if (i, j) in cellset else -1 for j in range(y0, y1 + 1)] for i in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': pix}


def extract(w):
    g, owner = render(w)
    out = []
    n = min(w['B'], 64)
    if w['ctr'] is not None:
        c = dict(w['ctr'])
        c.update(y=64 - n, h=n)
        out.append(c)
    spent = 1 if n < 64 else 0
    for typ, prefix, tags, layer, col, extra in (
            ('wall', 'wall_h_', ['axis', 'horizontal'], 1, 10, 0),
            ('player', 'piece_', ['movable', 'black'], 4, 5, spent),
            ('reflection', 'reflection_', ['mirror', 'gray'], 3, 4, 0)):
        for k, comp in enumerate(comps(g, (col, 0) if col != 4 else (4,))):
            if any(g[p] == col for p in comp):
                out.append(make(prefix + str(k + extra), typ, tags, layer, comp, g))
    tcomps = []
    for k in range(len(w['targets'])):
        cs = {p for p, o in owner.items() if o == k and g[p] == 11}
        if cs:
            tcomps.append(cs)
    tcomps.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for k, cs in enumerate(tcomps):
        out.append(make('target_' + str(k), 'target', ['goal', 'yellow'], 2, cs, g))
    return out


def transition_function(state, action):
    if _mem['last'] is not None and canon(state) == _mem['last']:
        w = _mem['world']
    else:
        w = parse(state)
    w = step(w, action)
    out = extract(w)
    _mem['last'], _mem['world'] = canon(out), w
    return out
