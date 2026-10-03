# Mechanics: mirror-axis game on a 21x21 grid of 3x3 blocks (x = block row). Selection = axis (wall 0 dots) or a
# piece (0 hole centres); A1/A2 move it -/+1 row, A3/A4 move a piece -/+1 col (bounds, no piece overlap; the axis
# passes under pieces), A5 cycles axis -> pieces (size desc, min block) -> axis. Every piece block off the axis row
# reflects to row 2A-b (clipped, skipped on piece blocks). Render piece>reflection>target>wall per block (ring = top,
# centre = first solid below else top fill) and re-extract by colour. Unconfirmed: blocking rules for axis at edges.
import json

N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memo = {'out': None, 'model': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def comps(blocks):
    blocks, res = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (b[0] + d[0], b[1] + d[1])
                if nb in blocks:
                    blocks.remove(nb)
                    stack.append(nb)
        res.append(comp)
    return res


def parse(state):
    grid, others, axis, axis_sel, targets, pcells = {}, [], 0, False, [], {}
    for o in state:
        c = cells_of(o)
        grid.update({k: v for k, v in c.items()})
        t = o.get('type')
        if t == 'wall':
            ten = [k for k, v in c.items() if v == 10]
            if ten:
                axis = ten[0][0] // 3
            if any(v == 0 for v in c.values()):
                axis_sel = True
        elif t == 'target':
            targets.append({(k[0] // 3, k[1] // 3) for k in c})
        elif t == 'player':
            pcells.update(c)
        elif t != 'reflection':
            others.append(o)
    pblocks = {(k[0] // 3, k[1] // 3) for k, v in pcells.items() if v == 5 and k[0] % 3 == 0 and k[1] % 3 == 0}

    def centre(b):
        return grid.get((3 * b[0] + 1, 3 * b[1] + 1), -1)

    pieces, sel = [], 'axis'
    if axis_sel:
        pieces = comps(pblocks)
    else:
        zero = {b for b in pblocks if centre(b) == 0}
        amb = {b for b in pblocks if centre(b) not in (0, -1)}
        if zero:
            sb = set()
            for comp in comps(zero | amb):
                if comp & zero:
                    sb |= comp
        else:
            cand = [c for c in comps(pblocks) if all(b in amb for b in c)]
            sb = cand[0] if cand else set()
        if sb:
            pieces.append(sb)
            sel = 0
        pieces += comps(pblocks - sb)
    return {'axis': axis, 'sel': sel, 'pieces': [set(p) for p in pieces], 'targets': targets, 'others': others}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    if aid in DIRS:
        dr, dc = DIRS[aid]
        if m['sel'] == 'axis':
            if dc == 0 and 0 <= m['axis'] + dr < N:
                m['axis'] += dr
        else:
            k = m['sel']
            nb = {(r + dr, c + dc) for r, c in m['pieces'][k]}
            rest = set().union(*[p for i, p in enumerate(m['pieces']) if i != k])
            if all(0 <= r < N and 0 <= c < N for r, c in nb) and not (nb & rest):
                m['pieces'][k] = nb
    elif aid == 5 and m['pieces']:
        od = order(m['pieces'])
        if m['sel'] == 'axis':
            m['sel'] = od[0]
        else:
            pos = od.index(m['sel'])
            m['sel'] = od[pos + 1] if pos + 1 < len(od) else 'axis'
    return m


def render(m):
    A, stacks = m['axis'], {}
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    for i, p in enumerate(m['pieces']):
        for b in p:
            stacks.setdefault(b, []).append(('P', i))
    for r, c in sorted(allp):
        rr = 2 * A - r
        if r != A and 0 <= rr < N and (rr, c) not in allp:
            stacks.setdefault((rr, c), []).append(('R', 0))
    for i, t in enumerate(m['targets']):
        for b in t:
            stacks.setdefault(b, []).append(('T', i))
    for c in range(N):
        stacks.setdefault((A, c), []).append(('W', 0))
    ring_col = {'P': 5, 'R': 4, 'T': 11, 'W': 10}
    rank = {'P': 0, 'R': 1, 'T': 2, 'W': 3}
    col, owner = {}, {}
    for (r, c), st in stacks.items():
        st = sorted(set(st), key=lambda s: rank[s[0]])
        top = st[0]
        for dx in range(3):
            for dy in range(3):
                if dx != 1 or dy != 1:
                    col[(3 * r + dx, 3 * c + dy)] = ring_col[top[0]]
                    owner[(3 * r + dx, 3 * c + dy)] = top
        v, own = None, top
        for s in st:
            if s[0] == 'T':
                v, own = 11, s
                break
            if s[0] == 'W' and m['sel'] == 'axis':
                v, own = 0, s
                break
        if v is None:
            if top[0] == 'P':
                v = 0 if m['sel'] == top[1] else -1
            else:
                v = 4 if top[0] == 'R' else (-1 if top[0] == 'W' else 11)
        if v >= 0:
            col[(3 * r + 1, 3 * c + 1)] = v
            owner[(3 * r + 1, 3 * c + 1)] = own
    return col, owner


def cell_comps(cells):
    cells, res = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            x, y = stack.pop()
            comp.add((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        res.append(comp)
    return res


def make(typ, idx, cells, col):
    xs, ys = [k[0] for k in cells], [k[1] for k in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[col[(x0 + i, y0 + j)] if (x0 + i, y0 + j) in cells else -1 for j in range(h)] for i in range(w)]
    return {'name': PREFIX[typ] + str(idx), 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'tags': list(TAGS[typ]), 'pixels': pix}


def ranked(cs):
    return sorted(cs, key=lambda c: (min(k[0] for k in c), min(k[1] for k in c)))


def extract(m):
    col, owner = render(m)
    out = [dict(o) for o in m['others']]
    for typ, own in (('wall', 10), ('player', 5)):
        cs = ranked(cell_comps(k for k, v in col.items() if v in (own, 0)))
        for i, c in enumerate(cs):
            if any(col[k] == own for k in c):
                out.append(make(typ, i, c, col))
    for i, c in enumerate(ranked(cell_comps(k for k, v in col.items() if v == 4))):
        out.append(make('reflection', i, c, col))
    tc = []
    for t in range(len(m['targets'])):
        c = {k for k, v in col.items() if v == 11 and owner[k] == ('T', t)}
        if c:
            tc.append(c)
    for i, c in enumerate(ranked(tc)):
        out.append(make('target', i, c, col))
    return out


def copy_model(m):
    return {'axis': m['axis'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']],
            'targets': m['targets'], 'others': m['others']}


def transition_function(state, action):
    if _memo['out'] is not None and canon(state) == _memo['out']:
        m = copy_model(_memo['model'])
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m)
    _memo['out'], _memo['model'] = canon(out), copy_model(m)
    return out
