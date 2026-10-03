# Mechanics: one horizontal axis (wall row a, 3x3 blocks) + holed 3x3-block pieces; ACTION5 cycles selection axis -> pieces (largest first) -> axis.
# Axis: A1/A2 move it a row up/down (never blocked by pieces); A3/A4 no-op. Selected piece: A1-A4 move 3 cells, blocked by board edge / other pieces.
# Reflection: every piece block off the axis row mirrors to row 2a-b (both sides), skipping off-board dests and dests covered by any piece.
# Render layers wall<target<reflection<piece; per cell topmost solid else topmost hole fill (selected 0, reflection 4); re-extract 4-conn comps.
# Hypothesis kept unconfirmed: piece cycle order beyond 2 pieces; merged pieces split statelessly only via the selected piece's 0-centres.
import json

N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
_memo = {'canon': None, 'model': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells_of(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def blocks_of(o, want=None):
    return {(x // 3, y // 3) for x, y, v in cells_of(o) if want is None or v in want}


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        c = set(st)
        while st:
            bx, by = st.pop()
            for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                if nb in blocks:
                    blocks.discard(nb)
                    c.add(nb)
                    st.append(nb)
        out.append(c)
    return out


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    a = min(o['x'] for o in walls) // 3
    wall_sel = any(v == 0 for o in walls for _, _, v in cells_of(o))
    targets = [blocks_of(o) for o in sorted(state, key=lambda o: o['name']) if o['type'] == 'target']
    target_blocks = set().union(*targets) if targets else set()
    other = [o for o in state if o['type'] in ('counter',)]
    pieces, sel = [], 'axis'
    for o in sorted((o for o in state if o['type'] == 'player'), key=lambda o: (o['x'], o['y'])):
        blks = blocks_of(o, (5,))
        zero = {(x // 3, y // 3) for x, y, v in cells_of(o) if v == 0}
        if wall_sel or not zero:
            parts = comps(blks) if len(comps(blks)) > 1 else [blks]
            pieces.extend(parts)
            continue
        # selected piece: 0-centre blocks grown over blocks whose centre shows a lower object
        chosen, grow = set(zero & blks), True
        centre = {(x // 3, y // 3): v for x, y, v in cells_of(o) if x % 3 == 1 and y % 3 == 1}
        while grow:
            grow = False
            for b in blks - chosen:
                if b not in centre and b in target_blocks and any(
                        (b[0] + dx, b[1] + dy) in chosen for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    chosen.add(b)
                    grow = True
        sel = len(pieces)
        pieces.append(chosen)
        pieces.extend(comps(blks - chosen))
    return {'a': a, 'sel': sel, 'pieces': pieces, 'targets': targets, 'other': other}


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    m = dict(m)
    m['pieces'] = [set(p) for p in m['pieces']]
    aid = action['action_id'] if isinstance(action, dict) else action
    if aid == 5:
        order = cycle_order(m['pieces'])
        if m['sel'] == 'axis':
            m['sel'] = order[0] if order else 'axis'
        else:
            k = order.index(m['sel'])
            m['sel'] = order[k + 1] if k + 1 < len(order) else 'axis'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None:
        return m
    if m['sel'] == 'axis':
        if d[1] == 0 and 0 <= m['a'] + d[0] < N:
            m['a'] += d[0]
        return m
    i = m['sel']
    moved = {(bx + d[0], by + d[1]) for bx, by in m['pieces'][i]}
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    if all(0 <= bx < N and 0 <= by < N for bx, by in moved) and not (moved & others):
        m['pieces'][i] = moved
    return m


def reflection_blocks(m):
    a = m['a']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for bx, by in occ:
        if bx == a:
            continue
        r = 2 * a - bx
        if 0 <= r < N and (r, by) not in occ:
            out.add((r, by))
    return out


def render(m):
    # sprites: (layer, kind, id, blocks, ring colour, hole fill or None for solid)
    sprites = [(1, 'wall', 0, {(m['a'], by) for by in range(N)}, 10, 0 if m['sel'] == 'axis' else -1)]
    for k, t in enumerate(m['targets']):
        sprites.append((2, 'target', k, t, 11, None))
    sprites.append((3, 'reflection', 0, reflection_blocks(m), 4, 4))
    for k, p in enumerate(m['pieces']):
        sprites.append((4, 'player', k, p, 5, 0 if m['sel'] == k else -1))
    sprites.sort(key=lambda s: -s[0])
    grid = {}
    for X in range(N * 3):
        for Y in range(N * 3):
            b = (X // 3, Y // 3)
            hole = X % 3 == 1 and Y % 3 == 1
            stack = [s for s in sprites if b in s[3]]
            pick = next((s for s in stack if s[5] is None or not hole), None)
            val = None
            if pick:
                val = pick[4]
            else:
                pick = next((s for s in stack if s[5] >= 0), None)
                if pick:
                    val = pick[5]
            if pick:
                grid[(X, Y)] = (pick[1], pick[2], val)
    return grid


def cell_comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        c = set(st)
        while st:
            x, y = st.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in cells:
                    cells.discard(nb)
                    c.add(nb)
                    st.append(nb)
        out.append(c)
    return out


def make_obj(name, typ, tags, cells, grid):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[grid[(x0 + i, y0 + j)][2] if (x0 + i, y0 + j) in cells else -1 for j in range(h)] for i in range(w)]
    return {'name': name, 'type': typ, 'tags': tags, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'pixels': px}


def extract(m, grid):
    out = [dict(o) for o in m['other']]
    zeros = {c for c, g in grid.items() if g[2] == 0}
    for typ, prefix, tags in (('wall', 'wall_h', ['axis', 'horizontal']), ('player', 'piece', ['movable', 'black'])):
        own = {c for c, g in grid.items() if g[0] == typ and g[2] != 0}
        cs = sorted(cell_comps(own | zeros), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
        for r, c in enumerate(cs):
            if c & own:
                out.append(make_obj('%s_%d' % (prefix, r), typ, tags, c, grid))
    refl = {c for c, g in grid.items() if g[0] == 'reflection'}
    cs = sorted(cell_comps(refl), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for r, c in enumerate(cs):
        out.append(make_obj('reflection_%d' % r, 'reflection', ['mirror', 'gray'], c, grid))
    tcs = []
    for k in range(len(m['targets'])):
        c = {p for p, g in grid.items() if g[0] == 'target' and g[1] == k}
        if c:
            tcs.append(c)
    tcs.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for r, c in enumerate(tcs):
        out.append(make_obj('target_%d' % r, 'target', ['goal', 'yellow'], c, grid))
    return out


def transition_function(state, action):
    if _memo['canon'] is not None and canon(state) == _memo['canon']:
        m = _memo['model']
    else:
        m = parse(state)
    m2 = step(m, action)
    out = extract(m2, render(m2))
    _memo['canon'] = canon(out)
    _memo['model'] = m2
    return out
