# Mechanics: two mirror axes on a 21x21 grid of 3x3 blocks (h = wall row, v = wall column; v drawn above h).
# ACTION5 cycles selection h -> v -> pieces (largest first, ties by min block) -> h; ACTION1-4 move the
# selection one block (x = row): axes only along their normal, pieces blocked by edges/pieces/axes.
# Render wall1(h<v) < target2 < reflection3 < piece4, holes transparent (selected 0, reflection 4); re-extract.
# Hypotheses: only v-axis reflections are drawn; piece cycle order by size; names rank among 0 pixels of other types.
import json

CELL, N = 3, 21
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'h': ['axis', 'horizontal'], 'v': ['axis', 'vertical'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
_memo = {}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def obj_pixels(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def block_of(p):
    return (p[0] // CELL, p[1] // CELL)


def centre(b):
    return (CELL * b[0] + 1, CELL * b[1] + 1)


def components(points):
    points, comps = set(points), []
    while points:
        stack, comp = [points.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for dx, dy in NB:
                q = (p[0] + dx, p[1] + dy)
                if q in points:
                    points.remove(q)
                    stack.append(q)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------- model rules per type

def axis_blocks(kind, pos):
    """Blocks of a full-length axis wall: h = block row pos, v = block column pos."""
    if pos is None:
        return set()
    return {(pos, k) for k in range(N)} if kind == 'h' else {(k, pos) for k in range(N)}


def piece_order(pieces):
    """Selection cycle order: larger pieces first, then by minimum block."""
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def cycle(model):
    """ACTION5: h -> v -> piece 0 .. last -> h (axes that do not exist are skipped)."""
    seq = [a for a in ('h', 'v') if model[a] is not None] + list(range(len(model['pieces'])))
    if not seq:
        return model
    i = seq.index(model['sel']) if model['sel'] in seq else -1
    return dict(model, sel=seq[(i + 1) % len(seq)])


def move_axis(model, kind, d):
    """An axis moves only along its normal; blocked by the board edge or a piece on the new line."""
    step = d[0] if kind == 'h' else d[1]
    if step == 0:
        return model
    na = model[kind] + step
    occupied = set().union(*model['pieces']) if model['pieces'] else set()
    if not 0 <= na < N or axis_blocks(kind, na) & occupied:
        return model
    return dict(model, **{kind: na})


def move_piece(model, k, d):
    """The selected piece shifts one block unless the edge, another piece or an axis wall blocks it."""
    pieces = model['pieces']
    moved = frozenset((bx + d[0], by + d[1]) for bx, by in pieces[k])
    rest = set().union(*[p for i, p in enumerate(pieces) if i != k]) if len(pieces) > 1 else set()
    walls = axis_blocks('h', model['h']) | axis_blocks('v', model['v'])
    if any(not (0 <= bx < N and 0 <= by < N) for bx, by in moved) or moved & rest or moved & walls:
        return model
    out = list(pieces)
    out[k] = moved
    return dict(model, pieces=out)


def step(model, action):
    a = action.get('action_id') if isinstance(action, dict) else action
    if a == 5:
        return cycle(model)
    d = DIRS.get(a)
    if d is None:
        return model
    if model['sel'] in ('h', 'v'):
        return move_axis(model, model['sel'], d)
    if isinstance(model['sel'], int):
        return move_piece(model, model['sel'], d)
    return model


def reflections(model):
    """Every piece block mirrored across the v axis (column c -> 2v - c); off-board blocks vanish."""
    if model['v'] is None:
        return set()
    out = set()
    for p in model['pieces']:
        out |= {(bx, 2 * model['v'] - by) for bx, by in p}
    return out


# ---------------------------------------------------------------- render + re-extract

def sprites(model):
    """Sprites low -> high: (kind, id, blocks, hole fill). fill 'solid' = no hole, None = transparent."""
    sp = []
    for a in ('h', 'v'):
        if model[a] is not None:
            sp.append(('wall', a, axis_blocks(a, model[a]), 0 if model['sel'] == a else None))
    for i, t in enumerate(model['targets']):
        sp.append(('target', i, set(t), 'solid'))
    sp.append(('reflection', 0, reflections(model), COLOR['reflection']))
    for i, p in enumerate(model['pieces']):
        sp.append(('player', i, set(p), 0 if model['sel'] == i else None))
    return sp


def composite(sp):
    """{pixel: (kind, id, value)}: topmost ring wins; a hole shows the lower solid pixel, else topmost fill."""
    by_block = {}
    for s in reversed(sp):
        for b in s[2]:
            if 0 <= b[0] < N and 0 <= b[1] < N:
                by_block.setdefault(b, []).append(s)
    frame = {}
    for b, stack in by_block.items():
        for i in range(CELL):
            for j in range(CELL):
                p = (CELL * b[0] + i, CELL * b[1] + j)
                hole = None
                for kind, sid, _, fill in stack:
                    if (i, j) != (1, 1) or fill == 'solid':
                        frame[p] = (kind, sid, COLOR[kind])
                        break
                    if fill is not None and hole is None:
                        hole = (kind, sid, fill)
                else:
                    if hole is not None:
                        frame[p] = hole
    return frame


def make_object(kind, pts, tags, layer):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y), v in pts.items():
        pix[x - x0][y - y0] = v
    return {'name': None, 'type': kind, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': layer, 'tags': list(tags), 'pixels': pix}


def extract(frame, model):
    """Walls/reflections/pieces = 4-connected components per type, targets one per sprite.
    A wall component takes prefix/tags of the axis owning most of its pixels. Wall and piece
    indices rank by (x, y) among their objects plus 0 pixels owned by other types."""
    layers = model['layers']
    by_kind, owner, tgt = {}, {}, {}
    for p, (kind, sid, v) in frame.items():
        if kind == 'target':
            tgt.setdefault(sid, {})[p] = v
        else:
            by_kind.setdefault(kind, {})[p] = v
            owner[p] = sid
    out = [make_object('target', pts, TAGS['target'], layers['target']) for pts in tgt.values()]
    for o in sorted(out, key=lambda o: (o['x'], o['y'])):
        o['name'] = 'target_' + str(out.index(o))
    for kind in ('wall', 'reflection', 'player'):
        pix = by_kind.get(kind, {})
        objs = []
        for comp in components(pix):
            pts = {p: pix[p] for p in comp}
            if kind == 'wall':
                ax = max(('v', 'h'), key=lambda a: sum(1 for p in comp if owner[p] == a))
                o = make_object(kind, pts, TAGS[ax], layers[kind])
                o['prefix'] = 'wall_' + ax + '_'
            else:
                o = make_object(kind, pts, TAGS[kind], layers[kind])
                o['prefix'] = 'reflection_' if kind == 'reflection' else 'piece_'
            objs.append(o)
        keys = [((o['x'], o['y']), o) for o in objs]
        if kind != 'reflection':
            keys += [(p, None) for p, (k, _, v) in frame.items() if v == 0 and k != kind]
        keys.sort(key=lambda t: t[0])
        for rank, (_, o) in enumerate(keys):
            if o is not None:
                o['name'] = o.pop('prefix') + str(rank)
        out += objs
    return out


# ---------------------------------------------------------------- stateless recovery

def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    wpix = {}
    for o in walls:
        wpix.update(obj_pixels(o))
    wblocks = {block_of(p) for p, v in wpix.items() if v == COLOR['wall']}
    rows, cols = {}, {}
    for bx, by in wblocks:
        rows[bx] = rows.get(bx, 0) + 1
        cols[by] = cols.get(by, 0) + 1
    h = max(rows, key=lambda r: (rows[r], -r)) if rows and max(rows.values()) > 2 else None
    v = max(cols, key=lambda c: (cols[c], -c)) if cols and max(cols.values()) > 2 else None
    zeros = {block_of(p) for p, val in wpix.items() if val == 0}
    sel = None
    if any(b[0] == h and b[1] != v for b in zeros):
        sel = 'h'
    elif zeros:
        sel = 'v'
    targets = [frozenset(block_of(p) for p in obj_pixels(o)) for o in state if o['type'] == 'target']
    lower = set()
    for o in state:
        if o['type'] in ('wall', 'target', 'reflection'):
            lower |= set(obj_pixels(o))
    pix = {}
    for o in state:
        if o['type'] == 'player':
            pix.update(obj_pixels(o))
    blocks = {block_of(p) for p, val in pix.items() if val == COLOR['player']}
    zero = {b for b in blocks if pix.get(centre(b)) == 0}
    amb = {b for b in blocks if centre(b) not in pix and centre(b) in lower}
    sel_blocks, stack = set(), list(zero)
    while stack:
        b = stack.pop()
        if b not in sel_blocks:
            sel_blocks.add(b)
            stack += [(b[0] + dx, b[1] + dy) for dx, dy in NB if (b[0] + dx, b[1] + dy) in amb]
    pieces = [frozenset(c) for c in components(blocks - sel_blocks)]
    if sel_blocks:
        pieces.append(frozenset(sel_blocks))
    pieces = piece_order(pieces)
    if sel is None and sel_blocks:
        sel = pieces.index(frozenset(sel_blocks))
    layers = dict(LAYER)
    for o in state:
        if o['type'] in layers:
            layers[o['type']] = o.get('layer', layers[o['type']])
    others = [dict(o) for o in state if o['type'] not in LAYER]
    return {'h': h, 'v': v, 'sel': sel, 'pieces': pieces, 'targets': targets,
            'layers': layers, 'others': others}


def render(model):
    return extract(composite(sprites(model)), model) + [dict(o) for o in model['others']]


def transition_function(state, action):
    global _memo
    model = _memo['model'] if _memo.get('out') == canon(state) else parse(state)
    new = step(model, action)
    out = render(new)
    _memo = {'out': canon(out), 'model': new}
    return out
