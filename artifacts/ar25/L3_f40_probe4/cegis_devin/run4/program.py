# Mechanics: one horizontal mirror axis (wall band on 3x3 block row A), movable 3x3-block pieces, static targets.
# ACTION5 cycles selection axis -> pieces by size desc (tie: min block) -> axis; ACTION1/2 move the selected axis
# or piece one block up/down, ACTION3/4 move a piece left/right (bounds 0..20, pieces may not overlap each other).
# Every piece block (r,c) off the axis reflects to (2A-r,c) unless that block holds a piece. Frame = layered render
# (piece>reflection>target>wall, holes transparent) then re-extraction by colour. Unconfirmed: axis/piece collisions.
import json

N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memo = {'out': None, 'model': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def frame_of(state):
    F = {}
    for o in state:
        for i, row in enumerate(o.get('pixels') or []):
            for j, v in enumerate(row):
                if v != -1:
                    F[(o['x'] + i, o['y'] + j)] = v
    return F


def block_comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for dr, dc in DIRS.values():
                n = (b[0] + dr, b[1] + dc)
                if n in blocks:
                    blocks.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(state):
    F = frame_of(state)
    walls = [o for o in state if o['type'] == 'wall']
    axis = walls[0]['x'] // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for row in o['pixels'] for v in row)
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append({(o['x'] + i) // 3 * 1000 + (o['y'] + j) // 3
                            for i, row in enumerate(o['pixels']) for j, v in enumerate(row) if v != -1})
    targets = [{(k // 1000, k % 1000) for k in t} for t in targets]
    pblocks = {(r, c) for r in range(N) for c in range(N) if F.get((3 * r, 3 * c)) == 5}

    def cls(b):
        v = F.get((3 * b[0] + 1, 3 * b[1] + 1), -1)
        if v == 0:
            return 'U' if axis_sel else 'S'
        return 'A' if v in (4, 11) else 'U'

    pieces, sel = [], 'axis' if axis_sel else None
    for comp in block_comps(pblocks):
        s = {b for b in comp if cls(b) == 'S'}
        if not s:
            pieces.append(frozenset(comp))
            continue
        grow = s | {b for b in comp if cls(b) == 'A'}
        selp = set().union(*[g for g in block_comps(grow) if g & s])
        sel = len(pieces)
        pieces.append(frozenset(selp))
        pieces.extend(frozenset(g) for g in block_comps(comp - selp))
    if sel is None:
        for i, p in enumerate(pieces):
            if all(cls(b) == 'A' for b in p):
                sel = i
                break
    sel_piece = pieces[sel] if isinstance(sel, int) else None
    return {'axis': axis, 'sel': sel_piece if sel_piece is not None else sel,
            'pieces': pieces, 'targets': targets}


def order(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def step(m, action):
    m = dict(m)
    pieces, sel = list(m['pieces']), m['sel']
    if action == 5:
        cyc = ['axis'] + order(pieces)
        m['sel'] = cyc[(cyc.index(sel) + 1) % len(cyc)] if sel in cyc else 'axis'
        return m
    if action not in DIRS or sel is None:
        return m
    dr, dc = DIRS[action]
    if sel == 'axis':
        if dc == 0 and 0 <= m['axis'] + dr < N:
            m['axis'] += dr
        return m
    moved = frozenset((r + dr, c + dc) for r, c in sel)
    others = set().union(*[p for p in pieces if p != sel]) if len(pieces) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
        m['pieces'] = [moved if p == sel else p for p in pieces]
        m['sel'] = moved
    return m


def render(m):
    A, sel = m['axis'], m['sel']
    stacks = {}
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    for i, p in enumerate(m['pieces']):
        for b in p:
            stacks.setdefault(b, []).append(('P', i, 5, False, 0 if p == sel else -1))
    for r, c in sorted(allp):
        rr = 2 * A - r
        if r != A and 0 <= rr < N and (rr, c) not in allp:
            st = stacks.setdefault((rr, c), [])
            if not any(s[0] == 'R' for s in st):
                st.append(('R', 0, 4, False, 4))
    for ti, t in enumerate(m['targets']):
        for b in t:
            stacks.setdefault(b, []).append(('T', ti, 11, True, 11))
    for c in range(N):
        stacks.setdefault((A, c), []).append(('W', 0, 10, False, 0 if sel == 'axis' else -1))
    F = {}
    for (r, c), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    F[(3 * r + i, 3 * c + j)] = (top[2], top[0], top[1])
        solid = [s for s in st if s[3]]
        fill = [s for s in st if s[4] >= 0]
        if solid:
            F[(3 * r + 1, 3 * c + 1)] = (solid[0][2], solid[0][0], solid[0][1])
        elif fill:
            F[(3 * r + 1, 3 * c + 1)] = (fill[0][4], fill[0][0], fill[0][1])
    return F


def cell_comps(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for dr, dc in DIRS.values():
                n = (p[0] + dr, p[1] + dc)
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        out.append(comp)
    return sorted(out, key=lambda comp: (min(p[0] for p in comp), min(p[1] for p in comp)))


def make_obj(name, typ, layer, tags, comp, F):
    x0, y0 = min(p[0] for p in comp), min(p[1] for p in comp)
    x1, y1 = max(p[0] for p in comp), max(p[1] for p in comp)
    pix = [[F[(x, y)][0] if (x, y) in comp else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
            'layer': layer, 'tags': tags, 'pixels': pix}


def extract(F, ntargets):
    out = []
    zeros = {p for p, v in F.items() if v[0] == 0}
    for col, typ, prefix, layer, tags in ((10, 'wall', 'wall_h_', 1, ['axis', 'horizontal']),
                                          (5, 'player', 'piece_', 4, ['movable', 'black'])):
        own = {p for p, v in F.items() if v[0] == col}
        for k, comp in enumerate(cell_comps(own | zeros)):
            if comp & own:
                out.append(make_obj(prefix + str(k), typ, layer, tags, comp, F))
    refl = {p for p, v in F.items() if v[0] == 4 and v[1] == 'R'}
    for k, comp in enumerate(cell_comps(refl)):
        out.append(make_obj('reflection_%d' % k, 'reflection', 3, ['mirror', 'gray'], comp, F))
    tcomps = []
    for ti in range(ntargets):
        comp = {p for p, v in F.items() if v[1] == 'T' and v[2] == ti}
        if comp:
            tcomps.append(comp)
    tcomps.sort(key=lambda comp: (min(p[0] for p in comp), min(p[1] for p in comp)))
    for k, comp in enumerate(tcomps):
        out.append(make_obj('target_%d' % k, 'target', 2, ['goal', 'yellow'], comp, F))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id', 6)
    key = canon(state)
    model = _memo['model'] if _memo['out'] == key else parse(state)
    new = step(model, action)
    out = [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    out += extract(render(new), len(new['targets']))
    _memo['out'], _memo['model'] = canon(out), new
    return out
