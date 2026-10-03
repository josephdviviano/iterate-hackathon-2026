# The board is a grid of 3x3 tiles with two movable mirror axes.
# Arrows move the selected axis or piece; edges and other pieces block pieces.
# Action 5 cycles horizontal axis, vertical axis, then pieces in reading order.
# Three gray images reflect across both axes; holes expose targets and axis dots.
# Effective actions tick HUD; no history needed; unobserved click/undo effects unconfirmed.

BACKGROUND, AXIS, TARGET, PIECE, IMAGE = 9, 10, 11, 5, 4


def neighbours(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        seed = min(remaining)
        group = {seed}
        remaining.remove(seed)
        queue = [seed]
        while queue:
            for q in neighbours(queue.pop()):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return sorted(groups, key=min)


def parse_board(frame):
    n = (len(frame) - 1) // 3
    rings = {(r, c): frame[3*r][3*c]
             for r in range(n) for c in range(n)}
    centres = {(r, c): frame[3*r+1][3*c+1]
               for r in range(n) for c in range(n)}
    wall = {q for q in rings if rings[q] == AXIS}
    horizontal = max(range(n), key=lambda r: sum(a == r for a, b in wall))
    vertical = max(range(n), key=lambda c: sum(b == c for a, b in wall))
    targets = {q for q in rings
               if any(frame[3*q[0]+dy][3*q[1]+dx] == TARGET
                      for dy in range(3) for dx in range(3))}
    cells = {q for q in rings if rings[q] == PIECE}
    h_dots = sum(r == horizontal and c != vertical and centres[r, c] == 0
                 for r, c in wall)
    v_dots = sum(c == vertical and r != horizontal and centres[r, c] == 0
                 for r, c in wall)
    selected_cells = set()
    if h_dots or v_dots:
        selected = 0 if h_dots > v_dots else 1
    else:
        selected_cells = {q for q in cells if centres[q] == 0}
        queue = list(selected_cells)
        while queue:
            for q in neighbours(queue.pop()):
                if q in cells and q not in selected_cells and centres[q] != BACKGROUND:
                    selected_cells.add(q)
                    queue.append(q)
        selected = 0
    pieces = components(cells - selected_cells)
    if selected_cells:
        pieces.append(selected_cells)
        pieces.sort(key=min)
        selected = 2 + pieces.index(selected_cells)
    return n, horizontal, vertical, pieces, targets, selected


def within_board(cells, n):
    return all(0 <= r < n and 0 <= c < n for r, c in cells)


def touches_other_piece(cells, pieces, index):
    return any(cells & p for i, p in enumerate(pieces) if i != index)


def update_wall(horizontal, vertical, selected, action, n):
    if selected == 0 and action in (1, 2):
        proposed = horizontal + (1 if action == 2 else -1)
        if 0 <= proposed < n:
            return proposed, vertical, True
    if selected == 1 and action in (3, 4):
        proposed = vertical + (1 if action == 4 else -1)
        if 0 <= proposed < n:
            return horizontal, proposed, True
    return horizontal, vertical, False


def update_player(pieces, index, action, n):
    if action not in (1, 2, 3, 4):
        return False
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    proposed = {(r + dr, c + dc) for r, c in pieces[index]}
    if not within_board(proposed, n) or touches_other_piece(proposed, pieces, index):
        return False
    pieces[index] = proposed
    return True


def update_reflection(pieces, horizontal, vertical, n):
    occupied = set().union(*pieces) if pieces else set()
    images = set()
    for r, c in occupied:
        for q in ((2*horizontal-r, c), (r, 2*vertical-c),
                  (2*horizontal-r, 2*vertical-c)):
            if q not in occupied and within_board((q,), n):
                images.add(q)
    return images


def update_counter(result, changed):
    if changed:
        column = len(result[0]) - 1
        for r in range(len(result) - 1):
            if result[r][column] != 12:
                result[r][column] = 12
                break


def render_board(frame, n, horizontal, vertical, pieces, targets, selected):
    result = [row[:] for row in frame]
    occupied = set().union(*pieces) if pieces else set()
    selected_piece = pieces[selected - 2] if selected >= 2 else set()
    images = update_reflection(pieces, horizontal, vertical, n)
    for r in range(n):
        for c in range(n):
            q = (r, c)
            axis_dot = (selected == 0 and r == horizontal or
                        selected == 1 and c == vertical)
            if q in occupied:
                ring = PIECE
            elif q in images:
                ring = IMAGE
            elif q in targets:
                ring = TARGET
            elif r == horizontal or c == vertical:
                ring = AXIS
            else:
                ring = BACKGROUND
            centre = (TARGET if q in targets else
                      0 if axis_dot or q in selected_piece else
                      IMAGE if q in images else BACKGROUND)
            for dy in range(3):
                for dx in range(3):
                    result[3*r+dy][3*c+dx] = centre if dy == dx == 1 else ring
    return result


def transition_function(state, action, frame):
    n, horizontal, vertical, pieces, targets, selected = parse_board(frame)
    aid = action.get('action_id') if isinstance(action, dict) else action
    changed = False
    if aid == 5:
        selected = (selected + 1) % (len(pieces) + 2)
        changed = True
    elif selected < 2:
        horizontal, vertical, changed = update_wall(horizontal, vertical, selected, aid, n)
    else:
        changed = update_player(pieces, selected - 2, aid, n)
    if not changed:
        return [row[:] for row in frame]
    result = render_board(frame, n, horizontal, vertical, pieces, targets, selected)
    update_counter(result, changed)
    return result
