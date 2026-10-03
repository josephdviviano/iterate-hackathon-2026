# Mechanics: arrows move the player 3 cells over floor colours {1,4}; a destination overlapping a token kicks it:
# the kicked chain slides 1 cell per tick (absorbing tokens it touches) over {1,4,15} until terrain, the portal
# bbox, the player or a non-fitting socket interior (inset 1) stops it; the player stays. Click on a token = the
# player takes its rect, the old player stays as an inert husk (centre -> 4). HUD row 63: ceil(n/2) zeros from the
# right, n = actions this level (continuity-gated, fallback 2*zeros). Unconfirmed: fitting-socket fill, A5/A7.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
STEP = 3
_mem = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def shift(r, d, k):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def inside(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def on_terrain(r, frame, covered, floor):
    return all(frame[j][i] in floor for i, j in cells(r) if (i, j) not in covered)


def socket_rejects(r, d, sockets):
    """Token may not enter a socket interior it does not fit; a fitting token stops once seated."""
    for s in sockets:
        inn = interior(s)
        if r == inn:
            return True
        nr = shift(r, d, 1)
        if overlap(nr, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def kick(blocks, start, d, player, frame, covered, bound, sockets):
    """Slide the chain started at `start` one cell per tick until any member is blocked."""
    blocks = list(blocks)
    for _ in range(64):
        chain = set(start)
        grew = True
        while grew:
            grew = False
            for k, b in enumerate(blocks):
                if k not in chain and any(overlap(shift(blocks[c], d, 1), b) for c in chain):
                    chain.add(k)
                    grew = True
        ok = True
        for c in chain:
            nr = shift(blocks[c], d, 1)
            if (not inside(nr, bound) or overlap(nr, player) or socket_rejects(blocks[c], d, sockets)
                    or not on_terrain(nr, frame, covered, BLOCK_FLOOR)):
                ok = False
                break
        if not ok:
            break
        for c in chain:
            blocks[c] = shift(blocks[c], d, 1)
    return blocks


def background(frame, covered, sockets, x, y):
    """Terrain under a mover: socket ring 4 / interior 1, else majority of nearest uncovered cells."""
    for s in sockets:
        if overlap((x, y, 1, 1), s):
            return 1 if overlap((x, y, 1, 1), interior(s)) else 4
    votes = []
    for dx, dy in DIRS.values():
        i, j = x + dx, y + dy
        while 0 <= i < 64 and 0 <= j < 63 and (i, j) in covered:
            i, j = i + dx, j + dy
        if 0 <= i < 64 and 0 <= j < 63:
            v = frame[j][i]
            votes.append(1 if v == 4 else v)
    return max(set(votes), key=votes.count) if votes else 1


def centre_idx(n):
    return range((n - 1) // 2, n // 2 + 1)


def draw_token(out, r, centre):
    x, y, w, h = r
    for i, j in cells(r):
        out[j][i] = 14
    for j in centre_idx(h):
        for i in centre_idx(w):
            out[y + j][x + i] = centre


def touching(a, b):
    """Sides of a touched by b: adjacent with perpendicular overlap."""
    sides = set()
    ox = a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
    oy = a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
    if ox and b[1] + b[3] == a[1]:
        sides.add("top")
    if ox and a[1] + a[3] == b[1]:
        sides.add("bottom")
    if oy and b[0] + b[2] == a[0]:
        sides.add("left")
    if oy and a[0] + a[2] == b[0]:
        sides.add("right")
    return sides


def draw_player(out, r, blocks):
    draw_token(out, r, 0)
    x, y, w, h = r
    sides = set()
    for b in blocks:
        sides |= touching(r, b)
    for i, j in cells(r):
        if ((j == y and "top" in sides) or (j == y + h - 1 and "bottom" in sides)
                or (i == x and "left" in sides) or (i == x + w - 1 and "right" in sides)):
            out[j][i] = 0


def hud_zeros(frame):
    return sum(1 for v in frame[63] if v == 0)


def transition_function(state, action, frame):
    player = next(rect(o) for o in state if o["type"] == "player")
    blocks = [rect(o) for o in state if o["type"] == "block"]
    sockets = [rect(o) for o in state if o["type"] == "target"]
    portals = [rect(o) for o in state if o["type"] == "portal"]
    bound = portals[0] if portals else (0, 0, 63, 63)
    covered = set(cells(player))
    for b in blocks:
        covered |= set(cells(b))

    n = _mem["n"] if _mem["frame"] == frame else 2 * hud_zeros(frame)
    out = [list(row) for row in frame]
    old_player, old_blocks = player, list(blocks)
    husk = None

    if isinstance(action, int) and action in DIRS:
        d = DIRS[action]
        nr = shift(player, d, STEP)
        hit = [k for k, b in enumerate(blocks) if overlap(nr, b)]
        if not inside(nr, bound):
            pass
        elif hit:
            blocks = kick(blocks, hit, d, player, frame, covered, bound, sockets)
        elif on_terrain(nr, frame, covered, PLAYER_FLOOR):
            player = nr
    elif isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        for k, b in enumerate(blocks):
            if overlap((cx, cy, 1, 1), b):
                husk, player = player, b
                blocks = blocks[:k] + blocks[k + 1:]
                break
    n += 1

    if husk is not None:
        x, y, w, h = husk
        for j in centre_idx(h):
            for i in centre_idx(w):
                out[y + j][x + i] = 4
        covered -= set(cells(husk))
    moved = [r for r in [old_player] + old_blocks if r not in blocks + [player] and r != husk]
    for r in moved:
        for i, j in cells(r):
            out[j][i] = background(frame, covered, sockets, i, j)
    for b in blocks:
        draw_token(out, b, 5)
    draw_player(out, player, blocks)

    z = (n + 1) // 2
    for i in range(64):
        out[63][i] = 0 if i >= 64 - z else frame[63][i] if frame[63][i] != 0 else 4
    _mem["frame"], _mem["n"] = out, n
    return out
