# Mirror-blocks/marker level, maze visible in frame (floor=5): 4x4 blocks (colour 10) move
# A1/A2 up/down 4, A3/A4 apart/together 4, iff dest 4x4 is all floor; armed arrows move the
# active marker's 4x4 cell (pos-1) the same way. Click 9 arms/switches (blocks->1, marker->11),
# click 1 disarms, else no-op; A5/A7 unobserved. HUD: colour-0 bars row 0 right + row 63 left,
# w=round(64a/150), a=calls since level start incl. no-ops, via output-frame continuity.

_MEMO = []  # list of (returned-frame key, action count a used), most recent last


def _key(frame):
    return tuple(tuple(r) for r in frame)


def _fill(frame, x, y, w, h, c):
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            frame[yy][xx] = c


def _free4(frame, x, y):
    if x < 0 or x + 3 > 63 or y < 1 or y + 3 > 62:
        return False
    for yy in range(y, y + 4):
        for xx in range(x, x + 4):
            if frame[yy][xx] != 5:
                return False
    return True


def _marker_at(markers, x, y):
    for m in markers:
        if m["x"] <= x < m["x"] + 2 and m["y"] <= y < m["y"] + 2:
            return m
    return None


def _bar_width(frame):
    w = 0
    for x in range(63, -1, -1):
        if frame[0][x] == 0:
            w += 1
        else:
            break
    return w


def _resume_counter(frame):
    k = _key(frame)
    for fk, fa in reversed(_MEMO):
        if fk == k:
            return fa
    w = _bar_width(frame)
    if w == 0:
        return 0
    return (300 * w + 149) // 128  # largest a with (128a+150)//300 == w


def transition_function(state, action, frame):
    out = [row[:] for row in frame]
    a = _resume_counter(frame) + 1

    blocks = [o for o in state if "block" in o.get("tags", ())]
    armed = [o for o in state if "armed" in o.get("tags", ())]
    markers = [o for o in state if o.get("type") == "marker"]
    active = [m for m in markers if "active" in m.get("tags", ())]

    if isinstance(action, dict):
        cx, cy = action.get("x", -1), action.get("y", -1)
        c = frame[cy][cx] if 0 <= cy < 64 and 0 <= cx < 64 else -1
        if c == 9:  # arm, or switch the active marker
            for b in blocks:
                _fill(out, b["x"], b["y"], 4, 4, 1)
            for m in active:
                _fill(out, m["x"], m["y"], 2, 2, 9)
            m = _marker_at(markers, cx, cy)
            if m is not None:
                _fill(out, m["x"], m["y"], 2, 2, 11)
        elif c == 1:  # disarm back to blocks
            for p in armed:
                _fill(out, p["x"], p["y"], 4, 4, 10)
            for m in active:
                _fill(out, m["x"], m["y"], 2, 2, 9)
    elif action in (1, 2, 3, 4):
        if armed:
            if active:
                m = active[0]
                dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
                cx0, cy0 = m["x"] - 1, m["y"] - 1
                nx, ny = cx0 + dx, cy0 + dy
                if _free4(frame, nx, ny):
                    _fill(out, m["x"], m["y"], 2, 2, 5)
                    _fill(out, nx + 1, ny + 1, 2, 2, 11)
        else:
            for i, b in enumerate(sorted(blocks, key=lambda o: o["x"])):
                if action == 1:
                    dx, dy = 0, -4
                elif action == 2:
                    dx, dy = 0, 4
                elif action == 3:
                    dx, dy = (-4, 0) if i == 0 else (4, 0)
                else:
                    dx, dy = (4, 0) if i == 0 else (-4, 0)
                nx, ny = b["x"] + dx, b["y"] + dy
                if _free4(frame, nx, ny):
                    _fill(out, b["x"], b["y"], 4, 4, 5)
                    _fill(out, nx, ny, 4, 4, 10)

    w = (128 * a + 150) // 300  # round(64a/150)
    for i in range(w):
        out[0][63 - i] = 0
        out[63][i] = 0

    _MEMO.append((_key(out), a))
    del _MEMO[:-16]
    return out
