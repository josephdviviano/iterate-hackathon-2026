# Mechanics: single horizontal mirror axis (wall row band) + holed 3x3-block pieces, gray reflections, yellow targets.
# Selection (axis / one piece / none) shows as 0-filled holes; A1/A2 move the selection by a block row (axis or piece),
# A3/A4 move a selected piece by a block column (blocked by bounds and other pieces; the axis passes under pieces).
# A5 cycles none/axis -> pieces (largest first, ties by bbox) -> axis. Every piece block off the axis row mirrors to 2A-r.
# Render piece>refl>target>wall, re-extract. Unconfirmed: A5 order largest-first vs nearest-axis (bbox fails step 64); merged unselected pieces unsplit.
N = 21
TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
        'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def comps(nodes):
    nodes, out = set(nodes), []
    while nodes:
        s = nodes.pop()
        st, c = [s], {s}
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in nodes:
                    nodes.discard(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    players = [o for o in state if o['type'] == 'player']
    targets = [o for o in state if o['type'] == 'target']
    axis = walls[0]['x'] // 3 if walls else None
    other = {}
    for o in state:
        if o['type'] != 'player':
            for r, c, v in cells(o):
                other[(r, c)] = v
    blocks, centre = set(), {}
    for o in players:
        px = {(r, c): v for r, c, v in cells(o)}
        for (r, c), v in px.items():
            if r % 3 == 0 and c % 3 == 0 and v == 5:
                b = (r // 3, c // 3)
                blocks.add(b)
                cv = px.get((r + 1, c + 1), -1)
                centre[b] = 'sel' if cv == 0 else ('amb' if (r + 1, c + 1) in other else 'empty')
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    pieces, sel = [], None
    for comp in comps(blocks):
        s = {b for b in comp if centre[b] == 'sel'} if not axis_sel else set()
        if s:
            grow = comps(s | {b for b in comp if centre[b] == 'amb'})
            s = set().union(*[g for g in grow if g & s])
            pieces.append(s)
            sel = len(pieces) - 1
            pieces.extend(comps(comp - s))
        else:
            pieces.append(comp)
    tgts = []
    for o in targets:
        tgts.append({(r // 3, c // 3) for r, c, _ in cells(o)})
    if axis_sel:
        sel = 'axis'
    return axis, pieces, sel, tgts


def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(axis, pieces, sel, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    if aid == 5:
        order = piece_order(pieces)
        if sel is None or not pieces:
            return axis, pieces, 'axis'
        if sel == 'axis':
            return axis, pieces, order[0]
        k = order.index(sel)
        return axis, pieces, order[k + 1] if k + 1 < len(order) else 'axis'
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None or sel is None:
        return axis, pieces, sel
    if sel == 'axis':
        na = axis + d[0]
        if d[0] and 0 <= na < N:
            axis = na
        return axis, pieces, sel
    moved = {(r + d[0], c + d[1]) for r, c in pieces[sel]}
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not moved & others:
        pieces = [moved if i == sel else p for i, p in enumerate(pieces)]
    return axis, pieces, sel


def render(axis, pieces, sel, tgts):
    # sprite stacks per block, top-down: (owner, ring, solid_centre, fill)
    stacks = {}
    pblocks = set().union(*pieces) if pieces else set()
    for i, p in enumerate(pieces):
        for b in p:
            stacks.setdefault(b, []).append((('player', i), 5, None, 0 if sel == i else -1))
    refl = set()
    if axis is not None:
        for r, c in pblocks:
            m = (2 * axis - r, c)
            if r != axis and 0 <= m[0] < N and m not in pblocks:
                refl.add(m)
    for b in sorted(refl):
        stacks.setdefault(b, []).append((('reflection', 0), 4, None, 4))
    for i, t in enumerate(tgts):
        for b in t:
            stacks.setdefault(b, []).append((('target', i), 11, 11, None))
    if axis is not None:
        for c in range(N):
            stacks.setdefault((axis, c), []).append((('wall', 0), 10, None, 0 if sel == 'axis' else -1))
    frame = {}
    for (br, bc), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    s = next((x for x in st if x[2] is not None), None)
                    if s:
                        frame[(3 * br + 1, 3 * bc + 1)] = (s[0], s[2])
                    else:
                        f = next((x for x in st if x[3] >= 0), None)
                        if f:
                            frame[(3 * br + 1, 3 * bc + 1)] = (f[0], f[3])
                else:
                    frame[(3 * br + i, 3 * bc + j)] = (top[0], top[1] if top[2] is None else top[2])
    return frame


def make(typ, idx, cs, frame):
    r0 = min(r for r, _ in cs); c0 = min(c for _, c in cs)
    r1 = max(r for r, _ in cs); c1 = max(c for _, c in cs)
    px = [[frame[(r, c)][1] if (r, c) in cs else -1 for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return {'name': PREFIX[typ] + str(idx), 'type': typ, 'tags': list(TAGS[typ]), 'x': r0, 'y': c0,
            'w': r1 - r0 + 1, 'h': c1 - c0 + 1, 'layer': LAYER[typ], 'pixels': px}


def bkey(cs):
    return (min(r for r, _ in cs), min(c for _, c in cs))


def extract(frame):
    out = []
    zeros = {p for p, (_, v) in frame.items() if v == 0}
    for typ in ('wall', 'player'):
        own = {p for p, (o, v) in frame.items() if o[0] == typ and v != 0}
        cl = sorted(comps(own | zeros), key=bkey)
        for k, cs in enumerate(cl):
            if cs & own:
                out.append(make(typ, k, cs, frame))
    refl = {p for p, (o, _) in frame.items() if o[0] == 'reflection'}
    for k, cs in enumerate(sorted(comps(refl), key=bkey)):
        out.append(make('reflection', k, cs, frame))
    tg = {}
    for p, (o, _) in frame.items():
        if o[0] == 'target':
            tg.setdefault(o[1], set()).add(p)
    for k, cs in enumerate(sorted(tg.values(), key=bkey)):
        out.append(make('target', k, cs, frame))
    return out


def transition_function(state, action):
    axis, pieces, sel, tgts = parse(state)
    axis, pieces, sel = step(axis, pieces, sel, action)
    out = extract(render(axis, pieces, sel, tgts))
    keep = [dict(o) for o in state if o['type'] not in PREFIX]
    for o in out:
        src = next((s for s in state if s['type'] == o['type']), None)
        if src is not None:
            o['tags'] = list(src['tags'])
            o['layer'] = src['layer']
    return keep + out
