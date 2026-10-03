# Mechanics: mirror-blocks game read/written on the frame. Floor = colour 5; walls 15/8 are drawn.
# Block mode (4x4 colour 10): A1/A2 move both blocks y-/+4, A3 left x-4 & right x+4, A4 the reverse;
# each block moves only if its target 4x4 is all floor. Click marker (2x2 colour 9) arms: blocks->1, marker->11.
# Armed: A1-4 move the active marker 4px if its target 4x4 cell is floor; click marker switches, click player disarms.
# HUD timer: w=3(a+1)//7 zeros at row 0 right and row 63 left; a = actions (hidden, continuity-gated). A5 unconfirmed in block mode.
FLOOR, BLOCK, PLAYER, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
_last = {"frame": None, "a": 0}


def _comps(frame, colours):
    seen, out = set(), []
    for y in range(64):
        for x in range(64):
            if frame[y][x] in colours and (x, y) not in seen:
                c, stack = frame[y][x], [(x, y)]
                seen.add((x, y))
                cells = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < 64 and 1 <= ny < 63 and (nx, ny) not in seen and frame[ny][nx] == c:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                out.append((c, min(p[0] for p in cells), min(p[1] for p in cells)))
    return out


def _bar_width(frame):
    w = 0
    while w < 64 and frame[0][63 - w] == BAR:
        w += 1
    return w


def _fill(frame, x, y, size, colour):
    for yy in range(y, y + size):
        for xx in range(x, x + size):
            frame[yy][xx] = colour


def _free(frame, x, y):
    if x < 0 or y < 1 or x + 4 > 64 or y + 4 > 63:
        return False
    return all(frame[yy][xx] == FLOOR for yy in range(y, y + 4) for xx in range(x, x + 4))


def _hit(px, py, x, y, size):
    return x <= px < x + size and y <= py < y + size


def transition_function(state, action, frame):
    if _last["frame"] is not None and frame == _last["frame"]:
        a = _last["a"]
    else:
        w = _bar_width(frame)
        a = 0 if w == 0 else (7 * w + 2) // 3
    out = [row[:] for row in frame]
    comps = _comps(frame, (BLOCK, PLAYER, MARK, ACTIVE))
    blocks = sorted((x, y) for c, x, y in comps if c == BLOCK)
    players = [(x, y) for c, x, y in comps if c == PLAYER]
    markers = [(x, y) for c, x, y in comps if c == MARK]
    active = [(x, y) for c, x, y in comps if c == ACTIVE]
    aid = action["action_id"] if isinstance(action, dict) else action
    moves = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
    if aid in moves:
        dx, dy = moves[aid]
        if blocks:
            deltas = [(dx, dy), (-dx, dy)] if len(blocks) == 2 else [(dx, dy)] * len(blocks)
            targets = []
            for (x, y), (ddx, ddy) in zip(blocks, deltas):
                targets.append((x + ddx, y + ddy) if _free(frame, x + ddx, y + ddy) else (x, y))
            for x, y in blocks:
                _fill(out, x, y, 4, FLOOR)
            for x, y in targets:
                _fill(out, x, y, 4, BLOCK)
        elif players and active:
            mx, my = active[0]
            cx, cy = mx - 1, my - 1
            tmp = [row[:] for row in frame]
            _fill(tmp, mx, my, 2, FLOOR)
            if _free(tmp, cx + dx, cy + dy):
                _fill(out, mx, my, 2, FLOOR)
                _fill(out, mx + dx, my + dy, 2, ACTIVE)
    elif aid == 6:
        px, py = action["x"], action["y"]
        hit_marker = next((m for m in markers if _hit(px, py, m[0], m[1], 2)), None)
        if blocks and hit_marker:
            for x, y in blocks:
                _fill(out, x, y, 4, PLAYER)
            _fill(out, hit_marker[0], hit_marker[1], 2, ACTIVE)
        elif players and hit_marker:
            for x, y in active:
                _fill(out, x, y, 2, MARK)
            _fill(out, hit_marker[0], hit_marker[1], 2, ACTIVE)
        elif players and any(_hit(px, py, x, y, 4) for x, y in players):
            for x, y in players:
                _fill(out, x, y, 4, BLOCK)
            for x, y in active:
                _fill(out, x, y, 2, MARK)
    a += 1
    w = min(64, 3 * (a + 1) // 7)
    for i in range(64):
        out[0][63 - i] = BAR if i < w else (frame[0][63 - i] if frame[0][63 - i] != BAR else FLOOR)
        out[63][i] = BAR if i < w else (frame[63][i] if frame[63][i] != BAR else FLOOR)
    _last["frame"], _last["a"] = [row[:] for row in out], a
    return out
