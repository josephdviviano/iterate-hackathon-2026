# Mechanics: one horizontal mirror axis (wall row of 3x3 blocks, 0 centre dots when selected) + pieces (3x3 blocks, holed) + gray reflections + yellow targets.
# ACTION5 cycles selection axis -> pieces (by min block) -> axis. Axis: A1/A2 move it one block row up/down; it passes UNDER pieces (not blocked by them).
# Selected piece: A1-A4 move it one block (x=row); blocked by board bounds and other pieces. Reflection block of (r,c) = (2*axis-r, c), clipped to the board.
# Render layers wall1<target2<reflection3<piece4; holes show the first lower solid, else the topmost sprite's fill; re-extract 4-conn comps (0 joins neighbours, 0-only comps count in ranks).
# Unconfirmed: piece entering the axis row (assumed allowed), A5 order after the first piece, splitting merged pieces when the axis is selected (hidden-state only).
import json

N = 21
TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
        'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h', 'player': 'piece', 'reflection': 'reflection', 'target': 'target'}
_memo = {'out': None, 'model': None}


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    players = [o for o in state if o['type'] == 'player']
    targets = sorted([o for o in state if o['type'] == 'target'], key=lambda o: (o['x'], o['y']))
    axis = min(o['x'] for o in walls) // 3
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    tblocks = [frozenset((r // 3, c // 3) for r, c, v in cells(o) if v == 11) for o in targets]
    tcells = {(r, c) for o in targets for r, c, v in cells(o)}
    pieces, sel = [], None
    for o in players:
        pix = {(r, c): v for r, c, v in cells(o)}
        blocks = {(r // 3, c // 3) for (r, c), v in pix.items() if v == 5}
        if axis_sel:
            pieces.append(blocks)
            continue
        zero = {b for b in blocks if pix.get((b[0] * 3 + 1, b[1] * 3 + 1)) == 0}
        if not zero:
            pieces.append(blocks)
            continue
        grow = set(zero)
        changed = True
        while changed:
            changed = False
            for b in blocks - grow:
                if (b[0] * 3 + 1, b[1] * 3 + 1) in tcells and any(
                        (b[0] + d, b[1] + e) in grow for d, e in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    grow.add(b)
                    changed = True
        sel_piece = grow
        rest = blocks - grow
        pieces.append(sel_piece)
        sel = sel_piece
        while rest:
            stack = [rest.pop()]
            comp = set(stack)
            while stack:
                b = stack.pop()
                for d, e in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (b[0] + d, b[1] + e)
                    if nb in rest:
                        rest.discard(nb)
                        comp.add(nb)
                        stack.append(nb)
            pieces.append(comp)
    pieces = sorted((frozenset(p) for p in pieces), key=min)
    seli = 'axis' if axis_sel else (pieces.index(frozenset(sel)) if sel is not None else 'axis')
    return {'axis': axis, 'sel': seli, 'pieces': pieces, 'targets': tblocks}


def step(m, action):
    m = dict(m)
    pieces = list(m['pieces'])
    if action == 5:
        if m['sel'] == 'axis':
            m['sel'] = 0 if pieces else 'axis'
        else:
            m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(pieces) else 'axis'
        return m
    if action not in (1, 2, 3, 4):
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if m['sel'] == 'axis':
        if dc == 0 and 0 <= m['axis'] + dr < N:
            m['axis'] += dr
        return m
    i = m['sel']
    moved = frozenset((r + dr, c + dc) for r, c in pieces[i])
    others = set().union(*[p for j, p in enumerate(pieces) if j != i]) if len(pieces) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
        pieces[i] = moved
    m['pieces'] = pieces
    return m


def render(m):
    # stack per cell: list of (layer, kind, solid, value, owner)
    stacks = {}

    def put(b, kind, frame, centre_solid, centre_val, owner=None):
        for i in range(3):
            for j in range(3):
                cell = (b[0] * 3 + i, b[1] * 3 + j)
                if i == 1 and j == 1:
                    ent = (LAYER[kind], kind, centre_solid, centre_val, owner)
                else:
                    ent = (LAYER[kind], kind, True, frame, owner)
                stacks.setdefault(cell, []).append(ent)

    a = m['axis']
    for c in range(N):
        if m['sel'] == 'axis':
            put((a, c), 'wall', 10, True, 0)
        else:
            put((a, c), 'wall', 10, False, None)
    for t, blocks in enumerate(m['targets']):
        for b in blocks:
            put(b, 'target', 11, True, 11, t)
    occupied = set()
    for i, p in enumerate(m['pieces']):
        put_sel = m['sel'] == i
        for b in p:
            put(b, 'player', 5, False, 0 if put_sel else None)
            occupied.add(b)
    refl = set()
    for p in m['pieces']:
        for r, c in p:
            rr = 2 * a - r
            if 0 <= rr < N:
                refl.add((rr, c))
    for b in refl:
        put(b, 'reflection', 4, False, 4)
    grid = {}
    for cell, st in stacks.items():
        st.sort(key=lambda e: -e[0])
        top = st[0]
        if top[2]:
            grid[cell] = (top[3], top[1], top[4])
            continue
        below = next((e for e in st[1:] if e[2]), None)
        if below is not None:
            grid[cell] = (below[3], below[1], below[4])
        elif top[3] is not None:
            grid[cell] = (top[3], top[1], top[4])
    return grid


def comps(cellset):
    seen, out = set(), []
    for c0 in sorted(cellset):
        if c0 in seen:
            continue
        seen.add(c0)
        stack, comp = [c0], [c0]
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cellset and nb not in seen:
                    seen.add(nb)
                    comp.append(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def make_obj(kind, idx, comp, grid):
    xs = [r for r, _ in comp]
    ys = [c for _, c in comp]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for r, c in comp:
        pix[r - x0][c - y0] = grid[(r, c)][0]
    return {'name': '%s_%d' % (PREFIX[kind], idx), 'type': kind, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[kind], 'tags': list(TAGS[kind]), 'pixels': pix}


def extract(grid, ntargets):
    out = []
    zeros = {c for c, v in grid.items() if v[0] == 0}
    for kind, col in (('wall', 10), ('player', 5)):
        own = {c for c, v in grid.items() if v[0] == col}
        cs = sorted(comps(own | zeros), key=lambda cp: (min(r for r, _ in cp), min(c for _, c in cp)))
        for idx, cp in enumerate(cs):
            if any(grid[c][0] == col for c in cp):
                out.append(make_obj(kind, idx, cp, grid))
    own = {c for c, v in grid.items() if v[0] == 4}
    cs = sorted(comps(own), key=lambda cp: (min(r for r, _ in cp), min(c for _, c in cp)))
    for idx, cp in enumerate(cs):
        out.append(make_obj('reflection', idx, cp, grid))
    tg = [[c for c, v in grid.items() if v[0] == 11 and v[2] == t] for t in range(ntargets)]
    tg = sorted([t for t in tg if t], key=lambda cp: (min(r for r, _ in cp), min(c for _, c in cp)))
    for idx, cp in enumerate(tg):
        out.append(make_obj('target', idx, cp, grid))
    return out


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    m = None
    if _memo['out'] is not None and canon(state) == _memo['out']:
        m = _memo['model']
    if m is None:
        m = parse(state)
    m2 = step(m, aid)
    out = [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    out += extract(render(m2), len(m2['targets']))
    _memo['out'] = canon(out)
    _memo['model'] = m2
    return out
