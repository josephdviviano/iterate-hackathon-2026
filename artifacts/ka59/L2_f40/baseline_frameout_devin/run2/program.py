# Mechanics: arrows move the player 3 cells over floor colours {1,4} inside the portal bounds; if its next rect hits tokens,
# they are kicked instead and slide 1 cell/tick as a push chain (contacted tokens join) over {1,4,15} until any member would
# leave bounds, hit a wall/husk, the player, or a socket interior (bbox inset 1) it does not exactly fit. Click on a token =
# player takes its rect; old player stays drawn as an inert husk (centre 4). Player shows 0 centre + 0 edge facing a token.
# HUD row 63: zeros from the right = ceil(actions/2) since level start (parity tracked via continuity). Unconfirmed: fit entry.
WALK_PLAYER = {1, 4}
WALK_BLOCK = {1, 4, 15}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_last = {"frame": None, "acts": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def cells(r):
    x, y, w, h = r
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def socket_interiors(state):
    return [(o["x"] + 1, o["y"] + 1, o["w"] - 2, o["h"] - 2) for o in state if o["type"] == "target"]


def in_socket_border(state, c):
    for o in state:
        if o["type"] == "target":
            r = rect(o)
            inner = (r[0] + 1, r[1] + 1, r[2] - 2, r[3] - 2)
            if overlap(r, (c[0], c[1], 1, 1)) and not overlap(inner, (c[0], c[1], 1, 1)):
                return True
    return False


def background(frame, state, movers):
    occ = set()
    for r in movers:
        occ.update(cells(r))
    bg = [row[:] for row in frame]
    for (x, y) in occ:
        if in_socket_border(state, (x, y)):
            bg[y][x] = 4
            continue
        if any(overlap(s, (x, y, 1, 1)) for s in socket_interiors(state)):
            bg[y][x] = 1
            continue
        guess = 1
        for d in ((1, 0), (0, 1)):
            sides = []
            for s in (1, -1):
                cx, cy = x, y
                while (cx, cy) in occ:
                    cx, cy = cx + d[0] * s, cy + d[1] * s
                if 0 <= cx < 64 and 0 <= cy < 64:
                    sides.append(frame[cy][cx])
            if len(sides) == 2 and sides[0] == sides[1] and sides[0] in WALK_BLOCK:
                guess = sides[0]
                break
        bg[y][x] = guess
    return bg


def free_for_block(r, bg, bounds, player, interiors):
    bx, by, bw, bh = bounds
    if r[0] < bx or r[1] < by or r[0] + r[2] > bx + bw or r[1] + r[3] > by + bh:
        return False
    if overlap(r, player):
        return False
    if any(bg[y][x] not in WALK_BLOCK for x, y in cells(r)):
        return False
    return all(not overlap(r, s) or (r[2], r[3]) == (s[2], s[3]) for s in interiors)


def free_for_player(r, bg, bounds):
    bx, by, bw, bh = bounds
    if r[0] < bx or r[1] < by or r[0] + r[2] > bx + bw or r[1] + r[3] > by + bh:
        return False
    return all(bg[y][x] in WALK_PLAYER for x, y in cells(r))


def slide(group, tokens, d, bg, bounds, player, interiors):
    tokens = list(tokens)
    group = set(group)
    while True:
        while True:
            grown = {j for j in range(len(tokens)) if j not in group
                     and any(overlap(shift(tokens[i], d), tokens[j]) for i in group)}
            if not grown:
                break
            group |= grown
        if not all(free_for_block(shift(tokens[i], d), bg, bounds, player, interiors) for i in group):
            return tokens
        for i in group:
            tokens[i] = shift(tokens[i], d)


def centre_idx(n):
    return range((n - 1) // 2, n // 2 + 1)


def paint_token(out, r, centre):
    x, y, w, h = r
    for cx, cy in cells(r):
        out[cy][cx] = 14
    for j in centre_idx(h):
        for i in centre_idx(w):
            out[y + j][x + i] = centre


def paint_player(out, r, tokens):
    paint_token(out, r, 0)
    x, y, w, h = r
    for d in DIRS.values():
        if any(overlap(shift(r, d), t) for t in tokens):
            for cx, cy in cells(r):
                if (d[0] == 1 and cx == x + w - 1) or (d[0] == -1 and cx == x) or \
                   (d[1] == 1 and cy == y + h - 1) or (d[1] == -1 and cy == y):
                    out[cy][cx] = 0


def hud_zeros(frame):
    z = 0
    while z < 64 and frame[63][63 - z] == 0:
        z += 1
    return z


def transition_function(state, action, frame):
    if _last["frame"] is not None and frame == _last["frame"]:
        acts = _last["acts"]
    else:
        acts = 2 * hud_zeros(frame)
    player = next(rect(o) for o in state if o["type"] == "player")
    tokens = [rect(o) for o in state if o["type"] == "block"]
    portal = next((rect(o) for o in state if o["type"] == "portal"), (0, 0, 63, 63))
    interiors = socket_interiors(state)
    bg = background(frame, state, [player] + tokens)
    husk = None
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            hit = [i for i, t in enumerate(tokens) if overlap(t, (action["x"], action["y"], 1, 1))]
            if hit:
                husk = player
                player = tokens.pop(hit[0])
    elif action in DIRS:
        d = DIRS[action]
        nxt = shift(player, d, 3)
        kicked = [i for i, t in enumerate(tokens) if overlap(nxt, t)]
        if kicked:
            tokens = slide(kicked, tokens, d, bg, portal, player, interiors)
        elif free_for_player(nxt, bg, portal):
            player = nxt
    out = [row[:] for row in bg]
    if husk is not None:
        paint_token(out, husk, 4)
    for t in tokens:
        paint_token(out, t, 5)
    paint_player(out, player, tokens)
    acts += 1
    for i in range(min(64, (acts + 1) // 2)):
        out[63][63 - i] = 0
    _last["frame"] = [row[:] for row in out]
    _last["acts"] = acts
    return out
