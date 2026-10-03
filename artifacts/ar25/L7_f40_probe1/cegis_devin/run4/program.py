# Mechanics: 3x3-block mirror game, x=row/y=col. H axis = wall row band ah, V axis = wall column av; selection
# (shown as 0 hole fills) cycles H -> V -> pieces by min block with ACTION5; A1/A2 move H or a piece by -/+1 block
# row, A3/A4 move V or a piece by -/+1 block col (bounded). Pieces cast a visible gray mirror across V and invisible
# (occluding) mirrors across H and both axes; a piece touching the H row casts an invisible V-mirror too (step 268).
# Frame rendered by per-block layer stacks, then re-extracted (4-conn comps, 0 cells rank with walls/pieces). Stateless.
import copy

N = 21
WALL, PIECE, REFL, TARG = 10, 5, 4, 11


def parse(state):
    g = {}
    for o in state:
        for i, row in enumerate(o.get('pixels') or []):
            for j, v in enumerate(row):
                if v != -1:
                    g[(o['x'] + i, o['y'] + j)] = v
    wb = {(r // 3, c // 3) for (r, c), v in g.items() if v == WALL}
    ah = max(range(N), key=lambda R: sum(1 for b in wb if b[0] == R))
    av = max(range(N), key=lambda C: sum(1 for b in wb if b[1] == C))
    pb = {(r // 3, c // 3) for (r, c), v in g.items() if v == PIECE}
    pieces = sorted(components(pb), key=min)
    targets = {(r // 3, c // 3) for (r, c), v in g.items() if v == TARG}
    zeros = {(R, C) for R in range(N) for C in range(N) if g.get((3 * R + 1, 3 * C + 1)) == 0}
    sel = None
    if any(R == ah and C != av and (R, C) not in pb for R, C in zeros):
        sel = 'H'
    elif any(C == av and R != ah and (R, C) not in pb for R, C in zeros):
        sel = 'V'
    else:
        for k, p in enumerate(pieces):
            if p & zeros:
                sel = k
                break
    if sel is None:
        sel = 'H' if (ah, av) in zeros else None
    return {'ah': ah, 'av': av, 'pieces': pieces, 'targets': targets, 'sel': sel}


def components(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    st.append(n)
        out.append(frozenset(comp))
    return out


def inb(b):
    return 0 <= b[0] < N and 0 <= b[1] < N


def step(m, action):
    sel, pieces = m['sel'], list(m['pieces'])
    if action == 5:
        order = ['H', 'V'] + list(range(len(pieces)))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return m
    if action not in (1, 2, 3, 4):
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if sel == 'H':
        if dr and 0 <= m['ah'] + dr < N:
            m['ah'] += dr
    elif sel == 'V':
        if dc and 0 <= m['av'] + dc < N:
            m['av'] += dc
    elif isinstance(sel, int):
        moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
        if all(inb(b) for b in moved):
            pieces[sel] = moved
            m['pieces'] = pieces
    return m


def mirrors(m):
    ah, av, pieces = m['ah'], m['av'], m['pieces']
    occ = set().union(*pieces) if pieces else set()
    vis, inv = set(), set()
    for p in pieces:
        touch = any(r == ah for r, c in p)
        for r, c in p:
            for b, visible in (((r, 2 * av - c), not touch), ((2 * ah - r, c), False),
                               ((2 * ah - r, 2 * av - c), False)):
                if inb(b) and b not in occ:
                    (vis if visible else inv).add(b)
    return vis, inv - vis


def render(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    vis, inv = mirrors(m)
    pown = {}
    for k, p in enumerate(m['pieces']):
        for b in p:
            pown[b] = k
    col, invc = {}, set()
    for R in range(N):
        for C in range(N):
            b = (R, C)
            stack = []  # (kind, ring colour, solid centre colour or None, hole fill)
            if b in pown:
                stack.append(('P', PIECE, None, 0 if sel == pown[b] else -1))
            if b in vis:
                stack.append(('R', REFL, None, REFL))
            if b in inv:
                stack.append(('I', -1, None, -1))
            if b in m['targets']:
                stack.append(('T', TARG, TARG, TARG))
            if C == av:
                stack.append(('W', WALL, 0 if sel == 'V' else None, -1))
            if R == ah:
                stack.append(('W', WALL, 0 if sel == 'H' else None, -1))
            if not stack:
                continue
            top = stack[0]
            solid = next((s[2] for s in stack if s[2] is not None), None)
            for i in range(3):
                for j in range(3):
                    cell = (3 * R + i, 3 * C + j)
                    if i == 1 and j == 1:
                        v = solid if solid is not None else top[3]
                    else:
                        v = top[1]
                    if top[0] == 'I' and v == -1:
                        invc.add(cell)
                    if v != -1:
                        col[cell] = v
    return col, invc


def cell_comps(cells):
    return sorted(components(cells), key=lambda s: (min(r for r, c in s), min(c for r, c in s)))


def make(name, typ, cells, value, layer, tags):
    r0 = min(r for r, c in cells); r1 = max(r for r, c in cells)
    c0 = min(c for r, c in cells); c1 = max(c for r, c in cells)
    px = [[value(r, c) if (r, c) in cells else -1 for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return {'name': name, 'type': typ, 'x': r0, 'y': c0, 'w': r1 - r0 + 1, 'h': c1 - c0 + 1,
            'layer': layer, 'tags': list(tags), 'pixels': px}


def gap(a, b):
    return max(max(b[0] - a[1], a[0] - b[1]), max(b[2] - a[3], a[2] - b[3])) - 1


def bbox(s):
    return (min(r for r, c in s), max(r for r, c in s), min(c for r, c in s), max(c for r, c in s))


def extract(col, invc):
    out = []
    zeros = {p for p, v in col.items() if v == 0}
    for colour, typ in ((WALL, 'wall'), (PIECE, 'player')):
        own = {p for p, v in col.items() if v == colour}
        for i, comp in enumerate(cell_comps(own | zeros)):
            if not comp & own:
                continue
            o = make('', typ, comp, lambda r, c: col[(r, c)], 1 if typ == 'wall' else 4, ())
            if typ == 'wall':
                vert = o['w'] >= o['h']
                o['name'] = ('wall_v_%d' if vert else 'wall_h_%d') % i
                o['tags'] = ['axis', 'vertical' if vert else 'horizontal']
            else:
                o['name'] = 'piece_%d' % i
                o['tags'] = ['movable', 'black']
            out.append(o)
    rc = {p for p, v in col.items() if v == REFL}
    for i, comp in enumerate(cell_comps(rc | invc)):
        vc = comp & rc
        if vc:
            out.append(make('reflection_%d' % i, 'reflection', vc, lambda r, c: REFL, 3, ('mirror', 'gray')))
    groups = [set(s) for s in components({p for p, v in col.items() if v == TARG})]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if gap(bbox(groups[a]), bbox(groups[b])) <= 3:
                    groups[a] |= groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    for i, g in enumerate(sorted(groups, key=lambda s: (bbox(s)[0], bbox(s)[2]))):
        out.append(make('target_%d' % i, 'target', g, lambda r, c: TARG, 2, ('goal', 'yellow')))
    return out


def transition_function(state, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    m = step(parse(state), aid)
    out = extract(*render(m))
    out += [copy.deepcopy(o) for o in state if o['type'] == 'counter']
    return out
