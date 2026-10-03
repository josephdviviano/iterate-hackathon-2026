# Mechanics: one horizontal mirror axis (wall, block row A) + 3x3-block pieces (player) + yellow targets; ACTION5 cycles
# selection axis -> pieces (largest first, then by min block) -> axis; A1/A2 move the selected axis/piece by one block row,
# A3/A4 move a selected piece by one block column (bounds 21x21 blocks, pieces may not overlap). Every piece block off the axis
# reflects to (2A-r, c) as a gray reflection unless a piece sits there. Frame = layered render (piece>refl>target>wall), then
# re-extract; wall/player names rank 4-conn comps of own cells + all 0 cells. Unconfirmed: true A5 piece order (sprite order?).
N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def cellmap(state):
    cells = {}
    for o in state:
        px = o.get('pixels')
        if not px:
            continue
        for i, row in enumerate(px):
            for j, v in enumerate(row):
                if v >= 0:
                    cells[(o['x'] + i, o['y'] + j)] = (v, o)
    return cells


def comps4(nodes):
    nodes, out = set(nodes), []
    while nodes:
        s = nodes.pop(); st, comp = [s], [s]
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in nodes:
                    nodes.remove(n); st.append(n); comp.append(n)
        out.append(comp)
    return out


def parse(state):
    cells = cellmap(state)
    walls = [o for o in state if o['type'] == 'wall']
    axis = walls[0]['x'] // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for row in o['pixels'] for v in row)
    pblocks = set()
    for (r, c), (v, o) in cells.items():
        if o['type'] == 'player' and r % 3 == 0 and c % 3 == 0 and v == 5:
            pblocks.add((r // 3, c // 3))
    groups = comps4(pblocks)
    pieces, sel = [], None

    def centre(b):
        return cells.get((3 * b[0] + 1, 3 * b[1] + 1))

    for g in groups:
        if axis_sel:
            pieces.append(set(g)); continue
        cls = {}
        for b in g:
            ce = centre(b)
            cls[b] = 'E' if ce is None else ('S' if ce[0] == 0 and ce[1]['type'] == 'player' else
                                             ('E' if ce[1]['type'] == 'player' else 'O'))
        sblocks = {b for b in g if cls[b] == 'S'}
        if sblocks:
            grown = set(sblocks)
            changed = True
            while changed:
                changed = False
                for b in g:
                    if b not in grown and cls[b] == 'O' and any(
                            (b[0] + dr, b[1] + dc) in grown for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        grown.add(b); changed = True
            for sc in comps4(grown):
                pieces.append(set(sc))
                if sel is None:
                    sel = len(pieces) - 1
            for rc in comps4(set(g) - grown):
                pieces.append(set(rc))
        else:
            pieces.append(set(g))
    if not axis_sel and sel is None:
        for i, p in enumerate(pieces):
            if all(centre(b) is not None and centre(b)[1]['type'] != 'player' for b in p):
                sel = i; break
    targets = []
    for o in state:
        if o['type'] == 'target':
            bl = set()
            for i, row in enumerate(o['pixels']):
                for j, v in enumerate(row):
                    if v >= 0:
                        bl.add(((o['x'] + i) // 3, (o['y'] + j) // 3))
            targets.append(bl)
    return {'axis': axis, 'axis_sel': axis_sel, 'pieces': pieces, 'sel': sel, 'targets': targets,
            'wall_proto': walls[0] if walls else None, 'others': [o for o in state if 'pixels' not in o],
            'protos': {o['type']: o for o in state}}


def order_key(p):
    return (-len(p), min(p))


def step(m, action):
    pieces = m['pieces']
    if action == 5:
        order = sorted(range(len(pieces)), key=lambda i: order_key(pieces[i]))
        if m['axis_sel']:
            if order:
                m['axis_sel'], m['sel'] = False, order[0]
        else:
            k = order.index(m['sel']) if m['sel'] in order else -1
            if k + 1 < len(order):
                m['sel'] = order[k + 1]
            else:
                m['axis_sel'], m['sel'] = True, None
        return
    if action not in DIRS:
        return
    dr, dc = DIRS[action]
    if m['axis_sel']:
        if dc == 0 and 0 <= m['axis'] + dr < N:
            m['axis'] += dr
        return
    if m['sel'] is None:
        return
    moved = {(r + dr, c + dc) for r, c in pieces[m['sel']]}
    if any(not (0 <= r < N and 0 <= c < N) for r, c in moved):
        return
    others = set().union(*[p for i, p in enumerate(pieces) if i != m['sel']]) if len(pieces) > 1 else set()
    if moved & others:
        return
    pieces[m['sel']] = moved


def render(m):
    A = m['axis']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    refl = set()
    for p in m['pieces']:
        for r, c in p:
            if r != A:
                rr = 2 * A - r
                if 0 <= rr < N and (rr, c) not in occ:
                    refl.add((rr, c))
    # sprite = (kind, id, blocks, ring colour, centre solid?, centre value/fill)
    sprites = []
    for i, p in enumerate(m['pieces']):
        sprites.append(('player', i, p, 5, False, 0 if (i == m['sel'] and not m['axis_sel']) else -1))
    sprites.append(('reflection', 0, refl, 4, False, 4))
    for i, t in enumerate(m['targets']):
        sprites.append(('target', i, t, 11, True, 11))
    if m['wall_proto'] is not None:
        sprites.append(('wall', 0, {(A, c) for c in range(N)}, 10, False, 0 if m['axis_sel'] else -1))
    blocks = {}
    for s in sprites:
        for b in s[2]:
            blocks.setdefault(b, []).append(s)
    frame = {}
    for (R, C), stack in blocks.items():
        top = stack[0]
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    frame[(3 * R + i, 3 * C + j)] = (top[3], top[0], top[1])
        sol = next((s for s in stack if s[4]), None)
        if sol is not None:
            frame[(3 * R + 1, 3 * C + 1)] = (sol[5], sol[0], sol[1])
        else:
            f = next((s for s in stack if s[5] >= 0), None)
            if f is not None:
                frame[(3 * R + 1, 3 * C + 1)] = (f[5], top[0], top[1])
    return frame


def make(proto, name, cells, frame):
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    x, y = min(rs), min(cs)
    w, h = max(rs) - x + 1, max(cs) - y + 1
    px = [[-1] * h for _ in range(w)]
    for r, c in cells:
        px[r - x][c - y] = frame[(r, c)][0]
    return {'name': name, 'type': proto['type'], 'tags': list(proto['tags']), 'x': x, 'y': y, 'w': w, 'h': h,
            'layer': proto['layer'], 'pixels': px}


def bbox_key(cells):
    return (min(r for r, c in cells), min(c for r, c in cells))


def extract(m, frame):
    out = [dict(o) for o in m['others']]
    zeros = {k for k, v in frame.items() if v[0] == 0}
    protos = m['protos']
    for kind, prefix in (('wall', 'wall_h'), ('player', 'piece')):
        if kind not in protos:
            continue
        own = {k for k, v in frame.items() if v[1] == kind}
        cs = sorted(comps4(own | zeros), key=bbox_key)
        for rank, comp in enumerate(cs):
            mine = [k for k in comp if k in own]
            if mine:
                out.append(make(protos[kind], '%s_%d' % (prefix, rank), mine, frame))
    if 'reflection' in protos:
        own = {k for k, v in frame.items() if v[1] == 'reflection'}
        for rank, comp in enumerate(sorted(comps4(own), key=bbox_key)):
            out.append(make(protos['reflection'], 'reflection_%d' % rank, comp, frame))
    elif any(v[1] == 'reflection' for v in frame.values()):
        proto = {'type': 'reflection', 'tags': ['mirror', 'gray'], 'layer': 3}
        own = {k for k, v in frame.items() if v[1] == 'reflection'}
        for rank, comp in enumerate(sorted(comps4(own), key=bbox_key)):
            out.append(make(proto, 'reflection_%d' % rank, comp, frame))
    tcells = {}
    for k, v in frame.items():
        if v[1] == 'target':
            tcells.setdefault(v[2], []).append(k)
    for rank, cl in enumerate(sorted(tcells.values(), key=bbox_key)):
        out.append(make(protos['target'], 'target_%d' % rank, cl, frame))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id', 6)
    m = parse(state)
    step(m, action)
    return extract(m, render(m))
