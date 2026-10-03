# Mechanics: single horizontal mirror axis (wall band, one 3x3-block row) + holed 3x3-block pieces + static targets.
# ACTION5 cycles selection axis -> pieces (largest first) -> axis; ACTION1/2 move selection x-3/x+3, ACTION3/4 piece y-3/y+3.
# Every piece block off the axis row mirrors to block row 2a-b (clipped to board) as a gray reflection. Frame is re-rendered by
# layers wall1<target2<reflection3<piece4 (holes transparent; fill 0 if selected, reflection 4) and re-extracted per type.
# No reflection on piece-occupied blocks. Unconfirmed: piece blocking vs other pieces/board only; names index offset by 0-pixels of other types (seen for walls only).
NB = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
_mem = {'out': None, 'model': None}


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def blocks_of(cs):
    return {(x // 3, y // 3) for (x, y) in cs}


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        c = set(st)
        while st:
            bx, by = st.pop()
            for n in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                if n in blocks:
                    blocks.discard(n); c.add(n); st.append(n)
        out.append(c)
    return out


def split_merged(cs, own):
    """Split one player component into pieces using its hole centres."""
    bl = blocks_of(cs)
    sel, amb, rest = set(), set(), set()
    for b in bl:
        c = (b[0] * 3 + 1, b[1] * 3 + 1)
        v = cs.get(c, -1)
        if v == 0:
            sel.add(b)
        elif own.get(c) is not None:
            amb.add(b)
        else:
            rest.add(b)
    grow = True
    while grow and sel:
        grow = False
        for b in list(amb):
            if any((b[0] + dx, b[1] + dy) in sel for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                sel.add(b); amb.discard(b); grow = True
    pieces = [c for c in comps(rest | amb)]
    return (sel if sel else None), pieces


def parse(state):
    walls = [o for o in state if o.get('type') == 'wall']
    players = [o for o in state if o.get('type') == 'player']
    targets = [o for o in state if o.get('type') == 'target']
    axis = min(o['x'] for o in walls) // 3 if walls else 0
    wall_sel = any(0 in r for o in walls for r in o['pixels'])
    lower = {}
    for o in targets + [o for o in state if o.get('type') == 'reflection']:
        lower.update(cells(o))
    pieces, sel = [], None
    for o in players:
        cs = cells(o)
        s, others = split_merged(cs, lower)
        if s and not wall_sel:
            sel = len(pieces)
            pieces.append(s)
        elif s:
            others = comps(blocks_of(cs))
        pieces.extend(others)
    tg = [blocks_of(cells(o)) for o in targets]
    if wall_sel or sel is None:
        sel = -1
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': tg}


def step(m, action):
    m = {'axis': m['axis'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']],
         'targets': m['targets']}
    a = action.get('action_id') if isinstance(action, dict) else action
    if a == 5:
        order = sorted(range(len(m['pieces'])), key=lambda i: (-len(m['pieces'][i]), min(m['pieces'][i])))
        if m['sel'] == -1:
            m['sel'] = order[0] if order else -1
        else:
            k = order.index(m['sel'])
            m['sel'] = order[k + 1] if k + 1 < len(order) else -1
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(a)
    if d is None:
        return m
    if m['sel'] == -1:
        if d[1] == 0 and 0 <= m['axis'] + d[0] < NB:
            m['axis'] += d[0]
        return m
    p = m['pieces'][m['sel']]
    np_ = {(bx + d[0], by + d[1]) for bx, by in p}
    if any(not (0 <= bx < NB and 0 <= by < NB) for bx, by in np_):
        return m
    occ = set()
    for i, q in enumerate(m['pieces']):
        if i != m['sel']:
            occ |= q
    if np_ & occ:
        return m
    m['pieces'][m['sel']] = np_
    return m


def render(m):
    """Return {cell: (colour, owner_key)}; owner_key = ('wall'|'player'|'reflection', None) or ('target', idx)."""
    a = m['axis']
    sprites = []  # (layer, key, blocks, fill)
    sprites.append((1, ('wall', None), {(a, by) for by in range(NB)}, 0 if m['sel'] == -1 else -1))
    for i, t in enumerate(m['targets']):
        sprites.append((2, ('target', i), t, None))
    refl, occ = set(), set().union(*m['pieces']) if m['pieces'] else set()
    for p in m['pieces']:
        for bx, by in p:
            if bx != a and 0 <= 2 * a - bx < NB and (2 * a - bx, by) not in occ:
                refl.add((2 * a - bx, by))
    sprites.append((3, ('reflection', None), refl, 4))
    for i, p in enumerate(m['pieces']):
        sprites.append((4, ('player', i), p, 0 if i == m['sel'] else -1))
    sprites.sort(key=lambda s: -s[0])
    frame = {}
    for x in range(63):
        for y in range(63):
            b, centre = (x // 3, y // 3), (x % 3 == 1 and y % 3 == 1)
            cov = [s for s in sprites if b in s[2]]
            if not cov:
                continue
            res = None
            for s in cov:
                if s[3] is None or not centre:
                    res = (COLOR[s[1][0]], s[1]); break
            if res is None:
                for s in cov:
                    if s[3] >= 0:
                        res = (s[3], cov[0][1]); break
            if res is not None:
                frame[(x, y)] = res
    return frame


def cell_comps(cs):
    cs, out = set(cs), []
    while cs:
        st = [cs.pop()]
        c = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cs:
                    cs.discard(n); c.add(n); st.append(n)
        out.append(c)
    return out


def make_obj(typ, cs, frame):
    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
    x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    pix = [[frame[(x0 + i, y0 + j)][0] if (x0 + i, y0 + j) in cs else -1 for j in range(h)] for i in range(w)]
    return {'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': LAYER[typ], 'pixels': pix}


def extract(frame, tags):
    by_type = {'wall': set(), 'player': set(), 'reflection': set()}
    tcells = {}
    for c, (col, key) in frame.items():
        if key[0] == 'target':
            tcells.setdefault(key[1], set()).add(c)
        else:
            by_type[key[0]].add(c)
    objs = {t: [make_obj(t, c, frame) for c in cell_comps(cs)] for t, cs in by_type.items()}
    objs['target'] = [make_obj('target', cs, frame) for _, cs in sorted(tcells.items())]
    zeros = {t: [c for c, (col, key) in frame.items() if col == 0 and key[0] != t] for t in ('wall', 'player')}
    out = []
    prefix = {'wall': 'wall_h_', 'player': 'piece_', 'reflection': 'reflection_', 'target': 'target_'}
    for t, lst in objs.items():
        keys = [(o['x'], o['y'], 0, k) for k, o in enumerate(lst)]
        keys += [(x, y, 1, -1) for x, y in zeros.get(t, [])]
        keys.sort()
        for rank, kk in enumerate(keys):
            if kk[3] >= 0:
                o = lst[kk[3]]
                o['name'] = prefix[t] + str(rank)
                o['tags'] = list(tags.get(t, TAGS[t]))
                out.append(o)
    return out


def transition_function(state, action):
    if _mem["out"] is not None and canon(state) == _mem['out']:
        m = _mem['model']
    else:
        m = parse(state)
    m2 = step(m, action)
    tags = {}
    for o in state:
        if o.get('type') in TAGS and 'tags' in o:
            tags.setdefault(o['type'], o['tags'])
    out = [dict(o) for o in state if o.get('type') not in TAGS]
    out += extract(render(m2), tags)
    _mem['out'], _mem['model'] = canon(out), m2
    return out
