# Mechanics implemented:
# - Two mirrored 4x4 block players: arrow actions move them (1/2 vertical, 3/4 mirrored
#   horizontal per half); a move is allowed only when every destination cell is colour 5.
# - Click on an inactive marker arms players (colour 1) and activates that marker (colour 11);
#   click on an armed player reverts to blocks (colour 10) and deactivates all markers.
# - In armed mode arrows move the 2x2 active marker with the same colour-5 blocking rule.
# - HUD: colour-0 bars row0 (right-anchored) and row63 (left-anchored), width ceil(3(n-1)/7)
#   where n = action index since level start; unverified: marker direction when x >= 32.

_last_frame = None
_n = 0


def _tags(o):
    return set(o.get("tags", []))


def _is_armed(o):
    return "armed" in _tags(o) or "color_1" in _tags(o)


def _is_active_marker(o):
    return "active" in _tags(o) or "color_11" in _tags(o)


def _player_color(o):
    if _is_armed(o):
        return 1
    if "cyan" in _tags(o):
        return 10
    pix = o.get("pixels")
    if pix:
        return pix[0][0]
    return 10


def _rect_cells(o):
    x, y, w, h = o.get("x", 0), o.get("y", 0), o.get("w", 1), o.get("h", 1)
    return x, y, w, h


def _free(frame, x, y, w, h):
    for yy in range(y, y + h):
        if yy < 0 or yy >= 64:
            return False
        for xx in range(x, x + w):
            if xx < 0 or xx >= 64 or frame[yy][xx] != 5:
                return False
    return True


def _draw(out, x, y, w, h, c):
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            if 0 <= yy < 64 and 0 <= xx < 64:
                out[yy][xx] = c


def transition_function(state, action, frame):
    global _last_frame, _n
    out = [r[:] for r in frame]

    players = [o for o in state if o.get("type") == "player" and o.get("visible", True)]
    markers = [o for o in state if o.get("type") == "marker" and o.get("visible", True)]
    armed = any(_is_armed(o) for o in players)
    bars = [o for o in state if o.get("type") == "wall" and "color_0" in _tags(o)]

    # HUD action counter (per level)
    if _last_frame is not None and frame == _last_frame:
        _n += 1
    else:
        if not bars:
            _n = 1
        else:
            w0 = max(o.get("w", 1) for o in bars)
            _n = (7 * (w0 - 1)) // 3 + 3
    width = (3 * (_n - 1) + 6) // 7
    for i in range(width):
        out[0][63 - i] = 0
        out[63][i] = 0

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action.get("x", 0), action.get("y", 0)

        def hit(o):
            x, y, w, h = _rect_cells(o)
            return x <= cx < x + w and y <= cy < y + h

        m = next((o for o in markers if hit(o)), None)
        p = next((o for o in players if hit(o)), None)
        if m is not None and not _is_active_marker(m):
            for o in markers:
                x, y, w, h = _rect_cells(o)
                _draw(out, x, y, w, h, 9)
            x, y, w, h = _rect_cells(m)
            _draw(out, x, y, w, h, 11)
            for o in players:
                x, y, w, h = _rect_cells(o)
                _draw(out, x, y, w, h, 1)
        elif p is not None and _is_armed(p):
            for o in players:
                x, y, w, h = _rect_cells(o)
                _draw(out, x, y, w, h, 10)
            for o in markers:
                x, y, w, h = _rect_cells(o)
                _draw(out, x, y, w, h, 9)
    elif isinstance(action, int) and action in (1, 2, 3, 4):
        if armed:
            m = next((o for o in markers if _is_active_marker(o)), None)
            if m is not None:
                dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
                x, y, w, h = _rect_cells(m)
                nx, ny = x + dx, y + dy
                if _free(frame, nx, ny, w, h):
                    _draw(out, x, y, w, h, 5)
                    _draw(out, nx, ny, w, h, 11)
        else:
            for o in players:
                x, y, w, h = _rect_cells(o)
                left_half = x < 32
                if action == 1:
                    dx, dy = 0, -4
                elif action == 2:
                    dx, dy = 0, 4
                elif action == 3:
                    dx, dy = (-4, 0) if left_half else (4, 0)
                else:
                    dx, dy = (4, 0) if left_half else (-4, 0)
                nx, ny = x + dx, y + dy
                if _free(frame, nx, ny, w, h):
                    _draw(out, x, y, w, h, 5)
                    _draw(out, nx, ny, w, h, _player_color(o))

    _last_frame = out
    return out
