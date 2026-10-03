# Mechanics: frame-level mirror-blocks/marker game. Floor = colour 5; any other colour (15 wall, 8 hazard, markers, players) blocks.
# Block mode (two cyan-10 4x4 blocks): A1/A2 move both y-4/y+4; A3 left x-4 & right x+4; A4 mirrored; each block moves independently if its target 4x4 is floor.
# Click inactive marker (9) -> blocks become armed players (1), marker turns active (11); clicking another 9 switches the active marker; clicking a player (1) disarms (1->10, 11->9); other clicks no-op.
# Armed mode: A1-A4 move the active 2x2 marker by 4 if its target 2x2 is floor. HUD: row 0 right-aligned and row 63 left-aligned 0 bars, width floor(3(k+1)/7), k = actions since level start.
# Unconfirmed: hazard (8) contact effects / death resets, A5 and A7 (treated as no-ops), whether markers check the full 4x4 cell.
FLOOR = 5
BLOCK, ARMED, INACTIVE, ACTIVE, BAR = 10, 1, 9, 11, 0
_last = {"frame": None, "k": 0}


def _comp(frame, x, y):
    c = frame[y][x]
    seen, todo = {(x, y)}, [(x, y)]
    while todo:
        cx, cy = todo.pop()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < 64 and 1 <= ny < 63 and (nx, ny) not in seen and frame[ny][nx] == c:
                seen.add((nx, ny))
                todo.append((nx, ny))
    return seen


def _comps(frame, colour):
    out, seen = [], set()
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                c = _comp(frame, x, y)
                seen |= c
                out.append(c)
    return out


def _bbox(cells):
    return min(x for x, _ in cells), min(y for _, y in cells)


def _try_move(frame, new, cells, dx, dy, colour):
    dest = {(x + dx, y + dy) for x, y in cells}
    for x, y in dest:
        if not (0 <= x < 64 and 1 <= y < 63):
            return
        if (x, y) not in cells and frame[y][x] != FLOOR:
            return
    for x, y in cells:
        new[y][x] = FLOOR
    for x, y in dest:
        new[y][x] = colour


def _recolour(new, cells, colour):
    for x, y in cells:
        new[y][x] = colour


def _bar_width(k):
    return 3 * (k + 1) // 7


def _actions_from_bar(frame):
    w = sum(1 for v in frame[0] if v == BAR)
    k = 0
    while _bar_width(k) < w:
        k += 1
    return k


def _draw_hud(new, k):
    w = min(64, _bar_width(k))
    for x in range(64):
        new[0][x] = BAR if x >= 64 - w else FLOOR
        new[63][x] = BAR if x < w else FLOOR


def _step_blocks(frame, new, aid):
    blocks = sorted(_comps(frame, BLOCK), key=lambda c: _bbox(c)[0])
    moves = {1: ((0, -4), (0, -4)), 2: ((0, 4), (0, 4)), 3: ((-4, 0), (4, 0)), 4: ((4, 0), (-4, 0))}
    if aid not in moves or len(blocks) != 2:
        return
    for cells, (dx, dy) in zip(blocks, moves[aid]):
        _try_move(frame, new, cells, dx, dy, BLOCK)


def _step_marker(frame, new, aid):
    moves = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
    act = _comps(frame, ACTIVE)
    if aid in moves and act:
        _try_move(frame, new, act[0], moves[aid][0], moves[aid][1], ACTIVE)


def _click(frame, new, x, y):
    if not (0 <= x < 64 and 1 <= y < 63):
        return
    c = frame[y][x]
    if c == INACTIVE:
        for comp in _comps(frame, ACTIVE):
            _recolour(new, comp, INACTIVE)
        _recolour(new, _comp(frame, x, y), ACTIVE)
        for comp in _comps(frame, BLOCK):
            _recolour(new, comp, ARMED)
    elif c == ARMED:
        for comp in _comps(frame, ACTIVE):
            _recolour(new, comp, INACTIVE)
        for comp in _comps(frame, ARMED):
            _recolour(new, comp, BLOCK)


def transition_function(state, action, frame=None):
    if frame is None:
        return state
    k = _last["k"] if frame == _last["frame"] else _actions_from_bar(frame)
    new = [row[:] for row in frame]
    armed = any(v == ARMED for row in frame[1:63] for v in row)
    if isinstance(action, dict):
        _click(frame, new, action.get("x", -1), action.get("y", -1))
    elif armed:
        _step_marker(frame, new, action)
    else:
        _step_blocks(frame, new, action)
    k += 1
    _draw_hud(new, k)
    _last["frame"] = [row[:] for row in new]
    _last["k"] = k
    return new
