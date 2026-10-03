# Mechanics: two-axis mirror puzzle on a 3x3-cell block grid (x=row). Axes H (wall row band) and V (wall column);
# ACTION5 cycles selection H -> V -> pieces (by min block) -> H; A1/A2 move the selection a block up/down, A3/A4
# left/right (H ignores A3/A4, V ignores A1/A2). Pieces mirror across V as visible gray reflections, across H and
# both axes as invisible occluders; a piece on the H row turns its V-mirror invisible too. Frame is re-rendered by
# layer stacks and re-extracted (components of own colour + 0 dots, ranked by bbox). Unconfirmed: piece blocking.
N = 21
BOARD = 63


def blk(r, c):
    return r // 3, c // 3


def parse(state):
    val, typ = {}, {}
    for o in state:
        p = o.get('pixels')
        if not p:
            continue
        for i, row in enumerate(p):
            for j, v in enumerate(row):
                if v != -1:
                    val[(o['x'] + i, o['y'] + j)] = v
                    typ[(o['x'] + i, o['y'] + j)] = o['type']
    rows, cols, pblocks, tblocks = {}, {}, set(), set()
    for (r, c), v in val.items():
        b = blk(r, c)
        if v == 10:
            rows.setdefault(b[0], set()).add(b)
            cols.setdefault(b[1], set()).add(b)
        elif v == 5:
            pblocks.add(b)
        elif v == 11:
            tblocks.add(b)
    ah = max(rows, key=lambda k: len(rows[k])) if rows else None
    av = max(cols, key=lambda k: len(cols[k])) if cols else None
    pieces = block_comps(pblocks)
    pieces.sort(key=lambda s: min(s))
    zeros = [blk(r, c) for (r, c), v in val.items() if v == 0 and r % 3 == 1 and c % 3 == 1]
    sel = None
    for b in zeros:
        if b in pblocks:
            continue
        if b[0] == ah and b[1] != av:
            sel = 'H'
        elif b[1] == av and b[0] != ah and sel is None:
            sel = 'V'
    if sel is None:
        for b in zeros:
            if b[0] == ah and b[1] == av and b not in pblocks:
                sel = 'H'
    if sel is None:
        for i, p in enumerate(pieces):
            if any(b in p for b in zeros):
                sel = i
                break
    return {'ah': ah, 'av': av, 'pieces': pieces, 'targets': tblocks, 'sel': sel}


def block_comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in blocks:
                    blocks.remove(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def inb(b):
    return 0 <= b[0] < N and 0 <= b[1] < N


def step(m, action):
    sel, pieces = m['sel'], m['pieces']
    if action == 5:
        order = ['H', 'V'] + list(range(len(pieces)))
        if sel in order:
            m['sel'] = order[(order.index(sel) + 1) % len(order)]
        return m
    if action not in DELTA:
        return m
    dr, dc = DELTA[action]
    if sel == 'H':
        if dr and m['ah'] is not None and 0 <= m['ah'] + dr < N:
            m['ah'] += dr
    elif sel == 'V':
        if dc and m['av'] is not None and 0 <= m['av'] + dc < N:
            m['av'] += dc
    elif isinstance(sel, int) and sel < len(pieces):
        moved = {(r + dr, c + dc) for r, c in pieces[sel]}
        others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
        if all(inb(b) for b in moved) and not (moved & others):
            pieces[sel] = moved
    return m


def render(m):
    ah, av, pieces, sel = m['ah'], m['av'], m['pieces'], m['sel']
    stacks = {}

    def push(b, spr):
        if inb(b):
            stacks.setdefault(b, []).append(spr)
    occupied = set().union(*pieces) if pieces else set()
    for i, p in enumerate(pieces):
        for b in p:
            push(b, ('P', 5, 0 if sel == i else None, False))
    vis, inv = [], []
    for p in pieces:
        on_h = any(r == ah for r, c in p)
        for r, c in p:
            if av is not None:
                (inv if on_h else vis).append((r, 2 * av - c))
            if ah is not None:
                inv.append((2 * ah - r, c))
            if ah is not None and av is not None:
                inv.append((2 * ah - r, 2 * av - c))
    for b in vis:
        if b not in occupied:
            push(b, ('R', 4, 4, False))
    for b in inv:
        if b not in occupied and b not in vis:
            push(b, ('I', -1, None, False))
    for b in m['targets']:
        push(b, ('T', 11, 11, True))
    for axis, on in (('V', lambda b: b[1] == av), ('H', lambda b: b[0] == ah)):
        for r in range(N):
            for c in range(N):
                if on((r, c)):
                    s = sel == axis or (sel in ('H', 'V') and r == ah and c == av)
                    push((r, c), ('W', 10, 0 if s else None, s))
    for b in stacks:
        uniq = []
        for s in stacks[b]:
            if s not in uniq:
                uniq.append(s)
        stacks[b] = uniq
    cells = {}
    for (br, bc), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                pos = (3 * br + i, 3 * bc + j)
                if (i, j) != (1, 1):
                    cells[pos] = (top[1], top[0])
                    continue
                solid = next((s for s in st if s[3]), None)
                if solid is not None:
                    cells[pos] = (solid[2], solid[0])
                elif top[2] is not None:
                    cells[pos] = (top[2], top[0])
                else:
                    cells[pos] = (-1, top[0])
    return {p: v for p, v in cells.items() if p[0] < BOARD and p[1] < BOARD}


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells:
                    cells.remove(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    out.sort(key=lambda s: (min(r for r, c in s), min(c for r, c in s)))
    return out


def make(name, typ, layer, tags, pts, colour):
    x0 = min(r for r, c in pts)
    y0 = min(c for r, c in pts)
    w = max(r for r, c in pts) - x0 + 1
    h = max(c for r, c in pts) - y0 + 1
    px = [[-1] * h for _ in range(w)]
    for (r, c) in pts:
        px[r - x0][c - y0] = colour[(r, c)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer,
            'tags': tags, 'pixels': px}


def gap(a, b):
    d = 0
    for k in (0, 1):
        amin, amax = min(p[k] for p in a), max(p[k] for p in a)
        bmin, bmax = min(p[k] for p in b), max(p[k] for p in b)
        d = max(d, bmin - amax, amin - bmax)
    return d


TGAP = 4


def extract(cells):
    col = {p: v for p, (v, o) in cells.items() if v != -1}
    objs = []
    zeros = {p for p, v in col.items() if v == 0}
    walls = {p for p, v in col.items() if v == 10}
    for i, cp in enumerate(comps(walls | zeros)):
        if cp & walls:
            w = max(r for r, c in cp) - min(r for r, c in cp) + 1
            h = max(c for r, c in cp) - min(c for r, c in cp) + 1
            d = 'horizontal' if h > w else 'vertical'
            objs.append(make('wall_%s_%d' % (d[0], i), 'wall', 1, ['axis', d], cp, col))
    pl = {p for p, v in col.items() if v == 5}
    for i, cp in enumerate(comps(pl | zeros)):
        if cp & pl:
            objs.append(make('piece_%d' % i, 'player', 4, ['movable', 'black'], cp, col))
    rcells = {p for p, (v, o) in cells.items() if o in ('R', 'I') and v in (-1, 4)}
    vis = {p for p in rcells if col.get(p) == 4}
    for i, cp in enumerate(comps(rcells)):
        if cp & vis:
            objs.append(make('reflection_%d' % i, 'reflection', 3, ['mirror', 'gray'], cp & vis, col))
    groups = comps({p for p, v in col.items() if v == 11})
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if gap(groups[a], groups[b]) <= TGAP:
                    groups[a] |= groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda s: (min(r for r, c in s), min(c for r, c in s)))
    for i, g in enumerate(groups):
        objs.append(make('target_%d' % i, 'target', 2, ['goal', 'yellow'], g, col))
    return objs


def transition_function(state, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    keep = [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    m = parse(state)
    if m['ah'] is None or m['av'] is None:
        return [dict(o) for o in state]
    m = step(m, aid)
    return keep + extract(render(m))
