# Mechanics: two mirror axes on a 3x3-cell block grid (x=row): H wall row ah, V wall column av, plus pieces.
# ACTION5 cycles selection H -> V -> pieces (by min block) -> H; A1/A2 move H or the piece by -/+1 block row,
# A3/A4 move V or the piece by -/+1 block col (axes: board bounds only; pieces: bounds + other pieces).
# Reflections: visible gray (r,2av-c); invisible occluders (2ah-r,c),(2ah-r,2av-c); none on piece blocks;
# render per-block stacks, re-extract by colour (targets merged while bbox gap<=3). Unconfirmed: piece on H row -> invisible V mirror.
N = 21
SOLID_T, HOLE = 's', 'h'


def blocks_of(o, colour):
    out = set()
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v == colour:
                out.add(((o['x'] + i) // 3, (o['y'] + j) // 3))
    return out


def cells_of(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def parse(state):
    wall, tgt, pieces, zero_wall, zero_piece = {}, set(), [], set(), set()
    for o in state:
        t = o['type']
        if t == 'wall':
            for b in blocks_of(o, 10):
                wall[b] = wall.get(b, 0) + 1
            for x, y, v in cells_of(o):
                if v == 0 and x % 3 == 1 and y % 3 == 1:
                    zero_wall.add((x // 3, y // 3))
        elif t == 'target':
            tgt |= blocks_of(o, 11)
        elif t == 'player':
            bl = set()
            for x, y, v in cells_of(o):
                if v == 5 and x % 3 == 0 and y % 3 == 0:
                    bl.add((x // 3, y // 3))
                if v == 0 and x % 3 == 1 and y % 3 == 1:
                    zero_piece.add((x // 3, y // 3))
            if bl:
                pieces.append(bl)
    rows, cols = [0] * N, [0] * N
    for (r, c) in wall:
        if 0 <= r < N and 0 <= c < N:
            rows[r] += 1
            cols[c] += 1
    ah = max(range(N), key=lambda r: rows[r])
    av = max(range(N), key=lambda c: cols[c])
    pieces.sort(key=min)
    sel = None
    if any(r == ah and c != av for r, c in zero_wall):
        sel = 'H'
    elif any(c == av and r != ah for r, c in zero_wall):
        sel = 'V'
    else:
        for k, p in enumerate(pieces):
            if p & zero_piece:
                sel = k
                break
    if sel is None:
        sel = 'H'
    return ah, av, sel, pieces, tgt


def step(ah, av, sel, pieces, action):
    if action == 5:
        order = ['H', 'V'] + list(range(len(pieces)))
        return ah, av, order[(order.index(sel) + 1) % len(order)], pieces
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return ah, av, sel, pieces
    if sel == 'H':
        if d[0] and 0 <= ah + d[0] < N:
            ah += d[0]
    elif sel == 'V':
        if d[1] and 0 <= av + d[1] < N:
            av += d[1]
    else:
        moved = {(r + d[0], c + d[1]) for r, c in pieces[sel]}
        others = set().union(*[p for k, p in enumerate(pieces) if k != sel])
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not moved & others:
            pieces = [moved if k == sel else p for k, p in enumerate(pieces)]
    return ah, av, sel, pieces


def render(ah, av, sel, pieces, tgt):
    stacks = {}

    def push(b, spr):
        if 0 <= b[0] < N and 0 <= b[1] < N:
            stacks.setdefault(b, []).append(spr)
    occupied = set().union(*pieces) if pieces else set()
    for k, p in enumerate(pieces):
        for b in p:
            push(b, ('P', 5, HOLE, 0 if sel == k else -1))
    vis, inv = set(), set()
    for p in pieces:
        on_h = any(r == ah for r, c in p)
        for r, c in p:
            (inv if on_h else vis).add((r, 2 * av - c))
            inv.add((2 * ah - r, c))
            inv.add((2 * ah - r, 2 * av - c))
    for b in vis - occupied:
        push(b, ('R', 4, HOLE, 4))
    for b in inv - occupied - vis:
        push(b, ('I', -1, HOLE, -1))
    for b in tgt:
        push(b, ('T', 11, SOLID_T, 11))
    for r in range(N):
        push((r, av), ('W', 10, SOLID_T if sel == 'V' else HOLE, 0 if sel == 'V' else -1))
    for c in range(N):
        push((ah, c), ('W', 10, SOLID_T if sel == 'H' else HOLE, 0 if sel == 'H' else -1))
    grid, kind = {}, {}
    for (r, c), st in stacks.items():
        top = st[0]
        centre = next((s[3] for s in st if s[2] == SOLID_T), top[3])
        for i in range(3):
            for j in range(3):
                cell = (3 * r + i, 3 * c + j)
                v = centre if (i, j) == (1, 1) else top[1]
                kind[cell] = top[0]
                if v != -1:
                    grid[cell] = v
    return grid, kind


def comps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, todo = {s}, [s]
        while todo:
            x, y = todo.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    todo.append(n)
        out.append(comp)
    return out


def bbox(cs):
    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
    return min(xs), min(ys), max(xs), max(ys)


def make(name, typ, tags, layer, cs, grid):
    x0, y0, x1, y1 = bbox(cs)
    pix = [[grid[(x, y)] if (x, y) in cs else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': pix}


def extract(grid, kind):
    out = []
    zeros = {c for c, v in grid.items() if v == 0}
    for colour, typ in ((10, 'wall'), (5, 'player')):
        cs = sorted(comps({c for c, v in grid.items() if v == colour} | zeros), key=lambda k: bbox(k)[:2])
        for i, k in enumerate(cs):
            if not any(grid[c] == colour for c in k):
                continue
            if typ == 'wall':
                x0, y0, x1, y1 = bbox(k)
                v = x1 - x0 >= y1 - y0
                out.append(make('wall_%s_%d' % ('v' if v else 'h', i), 'wall',
                                ['axis', 'vertical' if v else 'horizontal'], 1, k, grid))
            else:
                out.append(make('piece_%d' % i, 'player', ['movable', 'black'], 4, k, grid))
    refl = {c for c, v in grid.items() if v == 4} | {c for c, t in kind.items() if t == 'I'}
    cs = sorted(comps(refl), key=lambda k: bbox(k)[:2])
    for i, k in enumerate(cs):
        visc = {c for c in k if grid.get(c) == 4}
        if visc:
            out.append(make('reflection_%d' % i, 'reflection', ['mirror', 'gray'], 3, visc, grid))
    groups = [(set(k), bbox(k)) for k in comps({c for c, v in grid.items() if v == 11})]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = groups[a][1], groups[b][1]
                gx = max(A[0] - B[2], B[0] - A[2]) - 1
                gy = max(A[1] - B[3], B[1] - A[3]) - 1
                if gx <= 3 and gy <= 3:
                    s = groups[a][0] | groups[b][0]
                    groups[a] = (s, bbox(s))
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: g[1][:2])
    for i, (k, _) in enumerate(groups):
        out.append(make('target_%d' % i, 'target', ['goal', 'yellow'], 2, k, grid))
    return out


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    ah, av, sel, pieces, tgt = parse(state)
    ah, av, sel, pieces = step(ah, av, sel, pieces, aid)
    grid, kind = render(ah, av, sel, pieces, tgt)
    statics = [o for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    return statics + extract(grid, kind)
