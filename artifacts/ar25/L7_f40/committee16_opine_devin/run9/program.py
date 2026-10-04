# Geometry is a 21x21 board of 3x3 cells: axes 10, piece rings 5, targets 11.
# ACTION5 cycles horizontal axis, vertical axis, then pieces in reading order.
# Arrows move the selected axis or piece; edges and other pieces block pieces.
# Reflections across either axis and both are solid 4; targets show through holes.
# HUD advances only on effective actions; hidden history is unnecessary; clicks/undo unconfirmed.
N = 21
WALL, PLAYER, REFLECTION, TARGET, FLOOR, DOT, TICK = 10, 5, 4, 11, 9, 0, 12


def on_board(cell):
    return all(0 <= v < N for v in cell)


def touches_other_player(cells, others):
    return bool(cells & others)


def covered_by_player(cell, occupied):
    return cell in occupied


def target_visible_through_hole(cell, targets):
    return cell in targets


def selected_axis_at(cell, model):
    r, c = cell
    return ((model['selected'] == 'H' and r == model['H']) or
            (model['selected'] == 'V' and c == model['V']))


def parse_frame(frame):
    corners = [[frame[3*r][3*c] for c in range(N)] for r in range(N)]
    centres = [[frame[3*r+1][3*c+1] for c in range(N)] for r in range(N)]
    targets = {(r, c) for r in range(N) for c in range(N)
               if any(frame[3*r+i][3*c+j] == TARGET
                      for i in range(3) for j in range(3))}
    horizontal = max(range(N), key=lambda r: corners[r].count(WALL))
    vertical = max(range(N), key=lambda c: sum(corners[r][c] == WALL for r in range(N)))
    cells = {(r, c) for r in range(N) for c in range(N) if corners[r][c] == PLAYER}
    players = []
    while cells:
        start = min(cells)
        cells.remove(start)
        component, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for other in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if other in cells:
                    cells.remove(other)
                    component.add(other)
                    pending.append(other)
        players.append(component)
    players.sort(key=min)
    h_selected = any(corners[horizontal][c] == WALL and centres[horizontal][c] == DOT
                     for c in range(N) if c != vertical)
    v_selected = any(corners[r][vertical] == WALL and centres[r][vertical] == DOT
                     for r in range(N) if r != horizontal)
    selected = 'H' if h_selected else 'V' if v_selected else None
    if selected is None:
        selected = next((i for i, p in enumerate(players)
                         if any(centres[r][c] == DOT for r, c in p)), None)
    return {'H': horizontal, 'V': vertical, 'players': players,
            'targets': targets, 'selected': selected}


def update_selection(model):
    order = ['H', 'V'] + list(range(len(model['players'])))
    old = model['selected']
    index = order.index(old) if old in order else -1
    model['selected'] = order[(index+1) % len(order)]
    return old != model['selected']


def update_wall(model, delta):
    dr, dc = delta
    selected = model['selected']
    if selected == 'H':
        moved = model['H'] + dr
        if dc or not 0 <= moved < N:
            return False
        model['H'] = moved
    elif selected == 'V':
        moved = model['V'] + dc
        if dr or not 0 <= moved < N:
            return False
        model['V'] = moved
    else:
        return False
    return True


def update_player(model, delta):
    selected = model['selected']
    if not isinstance(selected, int):
        return False
    dr, dc = delta
    others = set().union(*(p for i, p in enumerate(model['players']) if i != selected))
    moved = {(r+dr, c+dc) for r, c in model['players'][selected]}
    if not all(on_board(cell) for cell in moved) or touches_other_player(moved, others):
        return False
    model['players'][selected] = moved
    return True


def update_target(model):
    return model['targets']


def update_reflection(model):
    horizontal, vertical = model['H'], model['V']
    occupied = set().union(*model['players'])
    reflected = set()
    for r, c in occupied:
        candidates = []
        if c != vertical:
            candidates.append((r, 2*vertical-c))
        if r != horizontal:
            candidates.append((2*horizontal-r, c))
        if r != horizontal and c != vertical:
            candidates.append((2*horizontal-r, 2*vertical-c))
        for cell in candidates:
            if on_board(cell) and not covered_by_player(cell, occupied):
                reflected.add(cell)
    return reflected


def render(model, frame):
    out = [row[:] for row in frame]
    occupied = set().union(*model['players'])
    selected = model['selected']
    active = model['players'][selected] if isinstance(selected, int) else set()
    targets = update_target(model)
    reflections = update_reflection(model)
    for r in range(N):
        for c in range(N):
            cell = r, c
            if covered_by_player(cell, occupied):
                ring = PLAYER
            elif cell in reflections:
                ring = REFLECTION
            elif cell in targets:
                ring = TARGET
            elif r == model['H'] or c == model['V']:
                ring = WALL
            else:
                ring = FLOOR
            if target_visible_through_hole(cell, targets):
                centre = TARGET
            elif selected_axis_at(cell, model) or cell in active:
                centre = DOT
            elif cell in reflections:
                centre = REFLECTION
            else:
                centre = FLOOR
            for i in range(3):
                for j in range(3):
                    out[3*r+i][3*c+j] = centre if i == j == 1 else ring
    return out


def update_counter(frame):
    for y in range(len(frame)):
        if frame[y][-1] != TICK:
            frame[y][-1] = TICK
            break


def transition_function(state, action, frame):
    model = parse_frame(frame)
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id == 5:
        changed = update_selection(model)
    else:
        delta = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action_id)
        changed = False
        if delta is not None:
            if model['selected'] in ('H', 'V'):
                changed = update_wall(model, delta)
            else:
                changed = update_player(model, delta)
    if not changed:
        return [row[:] for row in frame]
    out = render(model, frame)
    update_counter(out)
    return out
