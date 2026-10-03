"""Shared mechanisms of the mirror-axis grid game (3x3-pixel blocks on a 63x63 board).

Coordinates: pixels (x, y) are x-major (x = row, pixels[i][j] is at (o.x+i, o.y+j)).
Blocks are (bx, by) = (x // 3, y // 3). A model is a dict:
  axis     block row (horizontal axis) or column (vertical axis) of the mirror wall
  orient   'horizontal' | 'vertical'
  sel      -1 = axis selected, k >= 0 = pieces[k] selected
  pieces   list of frozensets of blocks, sorted by min block
  targets  list of frozensets of blocks (static)
  n        board size in blocks (21)
  others   objects passed through unchanged (counter HUD)
  templates first observed object per type (tags/layer source)
Standard library only; every function is pure (returns new values).
"""
import json

CELL = 3
N_BLOCKS = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
DEFAULT_TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
                'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---------------------------------------------------------------- helpers

def canon(state):
    """Order-independent fingerprint of a list of objects."""
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def obj_pixels(o):
    """{(x, y): value} for the visible (>= 0) pixels of an object."""
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def block_of(p):
    return (p[0] // CELL, p[1] // CELL)


def block_centre(b):
    return (CELL * b[0] + 1, CELL * b[1] + 1)


def components(points):
    """4-connected components of a set of grid points (pixels or blocks)."""
    points, comps = set(points), []
    while points:
        stack, comp = [points.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for dx, dy in NEIGHBOURS:
                q = (p[0] + dx, p[1] + dy)
                if q in points:
                    points.remove(q)
                    stack.append(q)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------- mechanisms

def action_direction(action):
    """action_decode: ACTION1/2/3/4 = one block up/down/left/right as (dbx, dby); else None.

    Accepts an int or {'action_id': ...}. ACTION5 is selection, ACTION6/7 are no-ops in
    the observed data (program 2 alternatively treats 6 as click-select, 7 as undo).
    """
    return DIRS.get(action_id(action))


def action_id(action):
    return action.get('action_id') if isinstance(action, dict) else action


def cycle_selection(sel, n_pieces):
    """selection_cycle: ACTION5 moves the selection axis(-1) -> piece 0 -> ... -> last -> axis.

    Pieces are indexed in (min block) order. Observed: axis -> first piece. Wrap past the
    last piece back to the axis is not exercised by the data (all programs assume it;
    program 6 falls back to piece 0 if there is no axis).
    """
    nxt = sel + 1
    return nxt if nxt < n_pieces else -1


def axis_blocks(axis, n=N_BLOCKS, orient='horizontal'):
    """Blocks covered by the full-length mirror wall."""
    if orient == 'horizontal':
        return {(axis, k) for k in range(n)}
    return {(k, axis) for k in range(n)}


def in_board(blocks, n=N_BLOCKS):
    return all(0 <= bx < n and 0 <= by < n for bx, by in blocks)


def move_piece(pieces, k, d, axis, n=N_BLOCKS, orient='horizontal', block_on_axis=True):
    """piece_move: the selected piece shifts one block by d unless blocked; returns new pieces.

    Blocked by the board edge, by another piece's blocks and (block_on_axis) by the axis
    wall. No blocked piece move occurs in the data; programs 1 and 4 omit the axis check
    (block_on_axis=False), program 6 checks pixel rows of the axis (equivalent).
    """
    moved = frozenset((bx + d[0], by + d[1]) for bx, by in pieces[k])
    rest = set().union(*[p for i, p in enumerate(pieces) if i != k])
    if not in_board(moved, n) or moved & rest:
        return list(pieces)
    if block_on_axis and moved & axis_blocks(axis, n, orient):
        return list(pieces)
    out = list(pieces)
    out[k] = moved
    return out


def move_axis(axis, d, pieces, n=N_BLOCKS, orient='horizontal', block_on_pieces=True):
    """axis_move: the selected axis moves one block along its normal only; returns new axis.

    Movement parallel to the wall is ignored (observed: ACTION3 with a horizontal axis =
    no change). Blocked by the board edge and (block_on_pieces) by any piece in the new
    row/column; never observed blocked, programs 1 and 4 check only bounds.
    """
    step = d[0] if orient == 'horizontal' else d[1]
    if step == 0:
        return axis
    na = axis + step
    if not 0 <= na < n:
        return axis
    if block_on_pieces and axis_blocks(na, n, orient) & set().union(*pieces):
        return axis
    return na


def reflect_blocks(blocks, axis, orient='horizontal'):
    """mirror_reflection: every piece block is mirrored across the axis block: r -> 2*axis - r.

    In pixels this is row x -> 2*(axis_px + 1) - x, i.e. block (2*axis_px)//3 - bx.
    Off-board reflections are clipped at render time.
    """
    if orient == 'horizontal':
        return {(2 * axis - bx, by) for bx, by in blocks}
    return {(bx, 2 * axis - by) for bx, by in blocks}


def hole_fill(kind, selected):
    """selection_marker: value shown in a block's centre hole when nothing lies beneath it.

    Selected piece/axis -> 0 (black dot); reflection -> 4 (gray, always); unselected
    piece/axis -> None (transparent, pixel unowned). Targets have solid centres.
    """
    if kind == 'reflection':
        return COLOR['reflection']
    if kind == 'target':
        return 'solid'
    return 0 if selected else None


def scene_sprites(model):
    """Sprites low -> high layer: wall(1) < targets(2) < reflection(3) < pieces(4)."""
    sp = [{'kind': 'wall', 'id': 0, 'blocks': axis_blocks(model['axis'], model['n'], model['orient']),
           'fill': hole_fill('wall', model['sel'] == -1)}]
    for i, t in enumerate(model['targets']):
        sp.append({'kind': 'target', 'id': i, 'blocks': set(t), 'fill': hole_fill('target', False)})
    refl = set()
    for p in model['pieces']:
        refl |= reflect_blocks(p, model['axis'], model['orient'])
    sp.append({'kind': 'reflection', 'id': 0, 'blocks': refl, 'fill': hole_fill('reflection', False)})
    for i, p in enumerate(model['pieces']):
        sp.append({'kind': 'player', 'id': i, 'blocks': set(p), 'fill': hole_fill('player', model['sel'] == i)})
    return sp


def composite(sprites, n=N_BLOCKS):
    """layered_composite: paint sprites into {pixel: (kind, id, value)} by layer, clipped to board.

    Ring pixels of the topmost covering sprite win. A centre hole is transparent: it shows
    the topmost lower solid pixel, else the fill of the topmost covering sprite with a
    fill (else unowned). Programs 2/6/7 paint low->high and fill a hole only if still
    empty (a lower hole fill then wins); the two differ only when holes stack, which the
    data never shows.
    """
    by_block = {}
    for s in reversed(sprites):
        for b in s['blocks']:
            if 0 <= b[0] < n and 0 <= b[1] < n:
                by_block.setdefault(b, []).append(s)
    frame = {}
    for b, stack in by_block.items():
        for i in range(CELL):
            for j in range(CELL):
                p = (CELL * b[0] + i, CELL * b[1] + j)
                centre = (i, j) == (1, 1)
                hole = None
                for s in stack:
                    if not centre or s['fill'] == 'solid':
                        frame[p] = (s['kind'], s['id'], COLOR[s['kind']])
                        break
                    if s['fill'] is not None and hole is None:
                        hole = (s['kind'], s['id'], s['fill'])
                else:
                    if hole is not None:
                        frame[p] = hole
    return frame


def wall_index_offset(frame):
    """wall_index_offset: wall names start at the number of black (0) pixels not owned by a wall.

    Observed names wall_h_{k + #0-dots of the selected piece}. Program 4 instead ranks walls
    among those 0 pixels by (x, y); identical here because the dots never sort after a wall.
    """
    return sum(1 for kind, _, v in frame.values() if v == 0 and kind != 'wall')


def make_object(kind, pts, name, templates=None):
    """Object dict with bbox, x-major pixels (-1 = unowned) and tags/layer from a template."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y), v in pts.items():
        pix[x - x0][y - y0] = v
    t = (templates or {}).get(kind, {})
    return {'name': name, 'type': kind, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': t.get('layer', LAYER[kind]), 'tags': list(t.get('tags', DEFAULT_TAGS[kind])),
            'pixels': pix}


def extract_objects(frame, orient='horizontal', templates=None):
    """component_reextraction: re-derive objects from the composited frame.

    Walls, pieces and reflections are 4-connected pixel components of their type (so
    touching pieces merge and occlusion splits the wall); targets stay one object per
    sprite. Each type is named prefix_i in (x, y) order of its visible bbox; walls are
    wall_h_/wall_v_ offset by wall_index_offset.
    """
    by_kind, by_target = {}, {}
    for p, (kind, sid, v) in frame.items():
        if kind == 'target':
            by_target.setdefault(sid, {})[p] = v
        else:
            by_kind.setdefault(kind, {})[p] = v
    groups = {'target': list(by_target.values())}
    for kind in ('wall', 'reflection', 'player'):
        pix = by_kind.get(kind, {})
        groups[kind] = [{p: pix[p] for p in comp} for comp in components(pix)]
    prefix = dict(PREFIX, wall='wall_h_' if orient == 'horizontal' else 'wall_v_')
    off = wall_index_offset(frame)
    out = []
    for kind, comps in groups.items():
        objs = sorted((make_object(kind, c, None, templates) for c in comps), key=lambda o: (o['x'], o['y']))
        for i, o in enumerate(objs):
            o['name'] = prefix[kind] + str(i + (off if kind == 'wall' else 0))
        out += objs
    return out


def static_passthrough(state):
    """static_passthrough: non-game objects (the counter HUD) are copied unchanged; it never ticks."""
    return [dict(o) for o in state if o['type'] not in LAYER]


def parse_model(state, n=N_BLOCKS):
    """stateless_recovery: rebuild the block model from an observed frame.

    Axis = min wall coordinate // 3 (orientation from the 'vertical' tag); axis selected
    iff a wall shows a 0. Targets = blocks of each target's visible pixels. Piece blocks
    come from player ring pixels; merged components are split by the centre hole:
    0 = selected, shows a lower object = ambiguous (joins an adjacent selected block),
    empty = unselected. With no 0 and no axis selection the selected piece is the one whose
    holes all show lower objects (else piece 0).
    """
    walls = [o for o in state if o['type'] == 'wall']
    orient = 'vertical' if any('vertical' in o.get('tags', []) for o in walls) else 'horizontal'
    axis = min((o['y'] if orient == 'vertical' else o['x']) for o in walls) // CELL if walls else n // 2
    axis_sel = any(v == 0 for o in walls for v in obj_pixels(o).values())
    targets = [frozenset(block_of(p) for p in obj_pixels(o)) for o in state if o['type'] == 'target']
    lower = set()
    for o in state:
        if o['type'] not in ('player', 'counter'):
            lower |= set(obj_pixels(o))
    pix = {}
    for o in state:
        if o['type'] == 'player':
            pix.update(obj_pixels(o))
    blocks = {block_of(p) for p, v in pix.items() if v == COLOR['player']}
    zero = {b for b in blocks if pix.get(block_centre(b)) == 0}
    amb = {b for b in blocks if block_centre(b) not in pix and block_centre(b) in lower}
    sel_blocks, stack = set(), list(zero)
    while stack:
        b = stack.pop()
        if b in sel_blocks:
            continue
        sel_blocks.add(b)
        stack += [(b[0] + dx, b[1] + dy) for dx, dy in NEIGHBOURS if (b[0] + dx, b[1] + dy) in amb]
    pieces = [frozenset(c) for c in components(blocks - sel_blocks)]
    if sel_blocks:
        pieces.append(frozenset(sel_blocks))
    pieces.sort(key=min)
    if axis_sel or not pieces:
        sel = -1
    elif sel_blocks:
        sel = pieces.index(frozenset(sel_blocks))
    else:
        sel = next((i for i, p in enumerate(pieces) if p <= amb), 0)
    templates = {}
    for o in state:
        templates.setdefault(o['type'], {'tags': o.get('tags'), 'layer': o.get('layer')})
    return {'axis': axis, 'orient': orient, 'sel': sel, 'pieces': pieces, 'targets': targets,
            'n': n, 'others': static_passthrough(state), 'templates': templates}


def recall(memo, state):
    """continuity_memo: reuse the hidden model if state is exactly the previous output, else None.

    Only matters where the frame is ambiguous (merged pieces, piece order); parse_model
    alone also reproduces the buffer.
    """
    if memo and memo.get('out') is not None and memo['out'] == canon(state):
        return memo['model']
    return None


def remember(out, model):
    """continuity_memo: new memo to store after producing `out` from `model`."""
    return {'out': canon(out), 'model': model}


def click_select(model, x, y):
    """click_select (program 2 only, unobserved): ACTION6 at a piece pixel selects it, on the axis selects the axis."""
    b = block_of((x, y))
    for i, p in enumerate(model['pieces']):
        if b in p:
            return dict(model, sel=i)
    if b in axis_blocks(model['axis'], model['n'], model['orient']):
        return dict(model, sel=-1)
    return model


def undo(history, model):
    """undo_history (program 2 only, unobserved): ACTION7 restores the previous model; returns (model, history)."""
    return (history[-1], history[:-1]) if history else (model, history)


def step(model, action, block_on_axis=True, block_on_pieces=True):
    """Glue: apply one action to a model (ACTION5 cycle, ACTION1-4 move, others no-op)."""
    a = action_id(action)
    if a == 5:
        return dict(model, sel=cycle_selection(model['sel'], len(model['pieces'])))
    d = action_direction(a)
    if d is None:
        return model
    if model['sel'] == -1:
        return dict(model, axis=move_axis(model['axis'], d, model['pieces'], model['n'],
                                          model['orient'], block_on_pieces))
    return dict(model, pieces=move_piece(model['pieces'], model['sel'], d, model['axis'], model['n'],
                                         model['orient'], block_on_axis))


def render_state(model):
    """Glue: model -> list of observed objects."""
    frame = composite(scene_sprites(model), model['n'])
    return extract_objects(frame, model['orient'], model['templates']) + [dict(o) for o in model['others']]
