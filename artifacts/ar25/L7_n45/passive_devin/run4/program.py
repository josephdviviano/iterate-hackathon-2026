# Mechanics: two mirror axes on a 21x21 grid of 3x3 blocks: row band R and column band C (walls), 2 holed pieces, targets.
# ACTION5 cycles selection C -> R -> pieces (by min block col,row) -> C; A1/A2 move C or the selected piece by -/+1 block
# column, A3/A4 move R or the selected piece by -/+1 block row (axes only bounded; pieces bounded and blocked by pieces).
# Visible gray reflections = piece blocks mirrored across R; across C / both = invisible occluders (rank only).
# Frame is re-rendered from layered block sprites and objects re-extracted; unconfirmed: cycle order with >2 pieces.
N = 21
NB = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def build_grid(state):
    col = [[None] * 63 for _ in range(63)]
    own = [[None] * 63 for _ in range(63)]
    for o in state:
        if 'pixels' not in o:
            continue
        for i, cl in enumerate(o['pixels']):
            for j, v in enumerate(cl):
                c, r = o['x'] + i, o['y'] + j
                if v != -1 and 0 <= r < 63 and 0 <= c < 63:
                    col[r][c] = v
                    own[r][c] = o['type']
    return col, own


def block_comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for dr, dc in NB:
                n = (r + dr, c + dc)
                if n in blocks:
                    blocks.discard(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def piece_key(p):
    return (min(c for r, c in p), min(r for r, c in p))


def parse(state, prev):
    col, own = build_grid(state)
    rows, cols = {}, {}
    pblocks, tblocks = set(), set()
    for r in range(63):
        for c in range(63):
            b = (r // 3, c // 3)
            if own[r][c] == 'wall':
                rows.setdefault(b[0], set()).add(b[1])
                cols.setdefault(b[1], set()).add(b[0])
            elif own[r][c] == 'target':
                tblocks.add(b)
            elif own[r][c] == 'player' and col[r][c] == 5:
                pblocks.add(b)
    R = max(rows, key=lambda k: len(rows[k])) if rows else 0
    C = max(cols, key=lambda k: len(cols[k])) if cols else 0

    def centre(b):
        return col[3 * b[0] + 1][3 * b[1] + 1], own[3 * b[0] + 1][3 * b[1] + 1]

    sel = None
    if any(centre((R, c)) == (0, 'wall') for c in range(N) if c != C):
        sel = 'R'
    elif any(centre((r, C)) == (0, 'wall') for r in range(N) if r != R):
        sel = 'C'
    elif centre((R, C)) == (0, 'wall'):
        sel = prev['sel'] if prev and prev['sel'] in ('R', 'C') else 'C'
    if prev and sorted(map(sorted, prev['pieces'])) == sorted(map(sorted, block_comps(pblocks))) \
            or prev and set().union(*prev['pieces']) == pblocks:
        pieces = [set(p) for p in prev['pieces']]
    else:
        pieces = []
        for comp in block_comps(pblocks):
            s0 = {b for b in comp if centre(b) == (0, 'player')}
            if not s0 or s0 == comp:
                pieces.append(comp)
                continue
            amb = {b for b in comp if centre(b)[1] not in (None, 'player')}
            grow = set(s0)
            ch = True
            while ch:
                ch = False
                for b in list(amb - grow):
                    if any((b[0] + dr, b[1] + dc) in grow for dr, dc in NB):
                        grow.add(b)
                        ch = True
            pieces.append(grow)
            pieces.extend(block_comps(comp - grow))
    pieces.sort(key=piece_key)
    if sel is None:
        for i, p in enumerate(pieces):
            if any(centre(b) == (0, 'player') for b in p):
                sel = i
        if sel is None:
            sel = prev['sel'] if prev else 'C'
    return {'R': R, 'C': C, 'sel': sel, 'pieces': pieces, 'targets': tblocks}


def step(m, action):
    m = dict(m)
    pieces = [set(p) for p in m['pieces']]
    sel = m['sel']
    if action == 5:
        if sel == 'C':
            sel = 'R'
        elif sel == 'R':
            sel = 0 if pieces else 'C'
        else:
            sel = sel + 1 if sel + 1 < len(pieces) else 'C'
    elif action in (1, 2, 3, 4):
        d = -1 if action in (1, 3) else 1
        dr, dc = (0, d) if action in (1, 2) else (d, 0)
        if sel == 'C' and dc:
            m['C'] = min(N - 1, max(0, m['C'] + dc))
        elif sel == 'R' and dr:
            m['R'] = min(N - 1, max(0, m['R'] + dr))
        elif isinstance(sel, int):
            moved = {(r + dr, c + dc) for r, c in pieces[sel]}
            others = set().union(set(), *[p for i, p in enumerate(pieces) if i != sel])
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
                pieces[sel] = moved
    m['sel'], m['pieces'] = sel, pieces
    return m


def render(m):
    R, C, sel, pieces = m['R'], m['C'], m['sel'], m['pieces']
    stacks = {}

    def push(b, spr):
        if 0 <= b[0] < N and 0 <= b[1] < N:
            stacks.setdefault(b, []).append(spr)
    allp = set().union(set(), *pieces)
    for i, p in enumerate(pieces):
        for b in p:
            push(b, ('player', 5, False, 0 if sel == i else None))
    for r, c in allp:
        if (2 * R - r, c) not in allp:
            push((2 * R - r, c), ('reflection', 4, False, 4))
    for r, c in allp:
        for b in ((r, 2 * C - c), (2 * R - r, 2 * C - c)):
            if b not in allp:
                push(b, ('inv', -1, False, None))
    for b in m['targets']:
        push(b, ('target', 11, True, 11))
    for k in range(N):
        for b in {(R, k), (k, C)}:
            if b == (R, C):
                s = sel in ('R', 'C')
            else:
                s = sel == ('R' if b[0] == R else 'C')
            if b not in stacks or stacks[b][-1][0] != 'wall':
                push(b, ('wall', 10, s, 0 if s else None))
    col = [[None] * 63 for _ in range(63)]
    own = [[None] * 63 for _ in range(63)]
    for (br, bc), st in stacks.items():
        top = st[0]
        for dr in range(3):
            for dc in range(3):
                r, c = 3 * br + dr, 3 * bc + dc
                if (dr, dc) != (1, 1):
                    col[r][c], own[r][c] = (top[1], top[0])
                    continue
                solid = [s for s in st if s[2]]
                if solid:
                    col[r][c], own[r][c] = solid[0][3], solid[0][0]
                elif top[3] is not None:
                    col[r][c], own[r][c] = top[3], top[0]
    return col, own


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for dr, dc in NB:
                n = (r + dr, c + dc)
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def bbox(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(cs), min(rs), max(cs), max(rs)


def make(name, typ, layer, tags, cells, col):
    x0, y0, x1, y1 = bbox(cells)
    px = [[col[y][x] if (y, x) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'layer': layer, 'tags': tags, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': px}


def extract(col, own):
    out = []
    zeros = {(r, c) for r in range(63) for c in range(63) if col[r][c] == 0}
    for typ, v, layer in (('wall', 10, 1), ('player', 5, 4)):
        own_cells = {(r, c) for r in range(63) for c in range(63) if own[r][c] == typ and col[r][c] == v}
        cs = sorted(comps(own_cells | zeros), key=lambda s: bbox(s)[:2])
        for i, s in enumerate(cs):
            if not (s & own_cells):
                continue
            x0, y0, x1, y1 = bbox(s)
            if typ == 'wall':
                vert = x1 - x0 >= y1 - y0
                out.append(make('wall_%s_%d' % ('v' if vert else 'h', i), 'wall', 1,
                                ['axis', 'vertical' if vert else 'horizontal'], s, col))
            else:
                out.append(make('piece_%d' % i, 'player', 4, ['movable', 'black'], s, col))
    rc = {(r, c) for r in range(63) for c in range(63) if own[r][c] in ('reflection', 'inv')}
    cs = sorted(comps(rc), key=lambda s: bbox(s)[:2])
    for i, s in enumerate(cs):
        vis = {p for p in s if own[p[0]][p[1]] == 'reflection'}
        if vis:
            out.append(make('reflection_%d' % i, 'reflection', 3, ['mirror', 'gray'], vis, col))
    groups = [s for s in comps({(r, c) for r in range(63) for c in range(63) if own[r][c] == 'target'})]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = bbox(groups[i]), bbox(groups[j])
                gx = max(b[0] - a[2], a[0] - b[2]) - 1
                gy = max(b[1] - a[3], a[1] - b[3]) - 1
                if max(gx, gy) <= 3:
                    groups[i] |= groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    for i, s in enumerate(sorted(groups, key=lambda s: bbox(s)[:2])):
        out.append(make('target_%d' % i, 'target', 2, ['goal', 'yellow'], s, col))
    return out


_memo = {'out': None, 'model': None}


def canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    prev = _memo['model'] if _memo['out'] is not None and canon(state) == _memo['out'] else None
    m = parse(state, prev)
    m2 = step(m, aid)
    out = extract(*render(m2))
    out += [dict(o) for o in state if 'pixels' not in o]
    _memo['out'], _memo['model'] = canon(out), m2
    return out
