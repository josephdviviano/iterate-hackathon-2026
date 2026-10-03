# Mechanics: model = {axis block-row a, pieces (3x3-block sets), targets (block sets), selection}; A1/A2 move the
# selected axis/piece one block up/down, A3/A4 move a selected piece left/right (axis ignores them); pieces are
# blocked by board bounds and other pieces, the axis is not (it passes under pieces); A5 cycles axis->pieces.
# Reflections = piece blocks mirrored across the axis (2a-r), clipped, skipped on piece blocks. Render layers
# wall<target<reflection<piece with hole centres, then re-extract 4-conn comps; hidden state only splits merged pieces.
N = 21
_memo = {'out': None, 'model': None}

def canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)

def cells(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v

def comps(cellset):
    seen, out = set(), []
    for c in sorted(cellset):
        if c in seen:
            continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cellset and n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(comp)
    return out

def block_groups(blocks):
    return [set(g) for g in comps(set(blocks))]

def parse(state):
    grid = {}
    for o in sorted(state, key=lambda o: o.get('layer', 0)):
        for x, y, v in cells(o):
            grid[(x, y)] = v
    walls = [o for o in state if o['type'] == 'wall']
    axis = min(o['x'] for o in walls) // 3 if walls else None
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append({(x // 3, y // 3) for x, y, v in cells(o) if v == 11})
    piece_blocks_all = set()
    raw = []
    for o in state:
        if o['type'] == 'player':
            bl = {(x // 3, y // 3) for x, y, v in cells(o) if v == 5}
            raw.append(bl)
            piece_blocks_all |= bl
    def centre(b):
        return grid.get((3 * b[0] + 1, 3 * b[1] + 1), -1)
    sel = None
    if axis is not None:
        for c in range(N):
            if (axis, c) not in piece_blocks_all and centre((axis, c)) == 0:
                sel = 'axis'
    pieces = []
    for bl in raw:
        on = {b for b in bl if centre(b) == 0 and not (sel == 'axis' and b[0] == axis)}
        amb = {b for b in bl if centre(b) not in (0, -1)}
        if sel == 'axis' or not on:
            pieces.append((bl, False))
            continue
        grown = set(on)
        changed = True
        while changed:
            changed = False
            for b in amb - grown:
                if any((b[0] + dx, b[1] + dy) in grown for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    grown.add(b)
                    changed = True
        pieces.append((grown, True))
        rest = bl - grown
        for g in block_groups(rest):
            pieces.append((g, False))
    plist = [p[0] for p in pieces]
    if sel != 'axis':
        sel = next((i for i, p in enumerate(pieces) if p[1]), None)
    return {'axis': axis, 'pieces': plist, 'targets': targets, 'sel': sel}

def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: min(pieces[i]))

def step(m, action):
    m = {'axis': m['axis'], 'pieces': [set(p) for p in m['pieces']],
         'targets': m['targets'], 'sel': m['sel']}
    if isinstance(action, dict):
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if action == 5:
        cyc = ['axis'] + order(m['pieces'])
        k = cyc.index(m['sel']) if m['sel'] in cyc else -1
        m['sel'] = cyc[(k + 1) % len(cyc)]
    elif action in d:
        dx, dy = d[action]
        if m['sel'] == 'axis':
            if dy == 0 and m['axis'] is not None and 0 <= m['axis'] + dx < N:
                m['axis'] += dx
        elif m['sel'] is not None:
            i = m['sel']
            moved = {(r + dx, c + dy) for r, c in m['pieces'][i]}
            others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i])
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
                m['pieces'][i] = moved
    return m

def reflections(m):
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    if m['axis'] is None:
        return out
    for p in m['pieces']:
        for r, c in p:
            rb = 2 * m['axis'] - r
            if 0 <= rb < N and (rb, c) not in occ:
                out.add((rb, c))
    return out

def render(m):
    stacks = {}
    def put(b, layer, owner, ring, centre_solid, fill):
        for i in range(3):
            for j in range(3):
                cell = (3 * b[0] + i, 3 * b[1] + j)
                if i == 1 and j == 1:
                    ent = (layer, owner, centre_solid, fill)
                else:
                    ent = (layer, owner, True, ring)
                stacks.setdefault(cell, []).append(ent)
    if m['axis'] is not None:
        s = m['sel'] == 'axis'
        for c in range(N):
            put((m['axis'], c), 1, ('wall', 0), 10, s, 0 if s else -1)
    for k, t in enumerate(m['targets']):
        for b in t:
            put(b, 2, ('target', k), 11, True, 11)
    for b in reflections(m):
        put(b, 3, ('refl', 0), 4, False, 4)
    for k, p in enumerate(m['pieces']):
        s = m['sel'] == k
        for b in p:
            put(b, 4, ('player', k), 5, False, 0 if s else -1)
    out = {}
    for cell, st in stacks.items():
        st.sort(key=lambda e: -e[0])
        ent = next((e for e in st if e[2]), None)
        if ent is not None:
            out[cell] = (ent[3], ent[1])
        elif st[0][3] >= 0:
            out[cell] = (st[0][3], st[0][1])
    return out

def mk(name, typ, comp, img, layer, tags):
    xs = [c[0] for c in comp]; ys = [c[1] for c in comp]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    cs = set(comp)
    pix = [[img[(x0 + i, y0 + j)][0] if (x0 + i, y0 + j) in cs else -1 for j in range(h)] for i in range(w)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer,
            'tags': list(tags), 'pixels': pix}

def ranked(comp_list):
    return sorted(comp_list, key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))

def extract(img, counter):
    out = [dict(counter)] if counter else []
    zeros = {c for c, v in img.items() if v[0] == 0}
    for typ, col, prefix, layer, tags in (('wall', 10, 'wall_h_', 1, ('axis', 'horizontal')),
                                          ('player', 5, 'piece_', 4, ('movable', 'black'))):
        own = {c for c, v in img.items() if v[0] == col}
        for k, comp in enumerate(ranked(comps(own | zeros))):
            if any(c in own for c in comp):
                out.append(mk(prefix + str(k), typ, comp, img, layer, tags))
    refl = {c for c, v in img.items() if v[0] == 4}
    for k, comp in enumerate(ranked(comps(refl))):
        out.append(mk('reflection_%d' % k, 'reflection', comp, img, 3, ('mirror', 'gray')))
    tg = {}
    for c, v in img.items():
        if v[1][0] == 'target' and v[0] == 11:
            tg.setdefault(v[1][1], []).append(c)
    for k, comp in enumerate(ranked(list(tg.values()))):
        out.append(mk('target_%d' % k, 'target', comp, img, 2, ('goal', 'yellow')))
    return out

def transition_function(state, action):
    counter = next((o for o in state if o['type'] == 'counter'), None)
    if _memo['out'] is not None and canon(state) == _memo['out']:
        model = _memo['model']
    else:
        model = parse(state)
    new = step(model, action)
    result = extract(render(new), counter)
    _memo['out'] = canon(result)
    _memo['model'] = new
    return result
