# Mechanics: player steps by its own 3-cell grid on frame floor {1,4}; colours 2/15 and the portal bound block it.
# player_kicks_block: if the next rect overlaps blocks they slide (1-cell ticks, push-chain closure) until a member
# hits colour 2, the bound, the player, or a socket interior (inset 1) it does not exactly fit; blocks cross 15.
# Click on a block: player possesses it (takes its rect), old player stays as inert husk (e rim, 4 centre).
# Render: e rim, centre 5 (block) / 0 (player), player sides touching a block all 0. HUD row 63: ceil(n/2) zeros, n = actions this level (continuity-gated; fallback n = 2*zeros). Unconfirmed: fitting-socket fill, husk collisions.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
TERRAIN = {1, 2, 15}
RIM, BLOCK_CORE, PLAYER_CORE, HUSK_CORE, HUD_BG, HUD_MARK = 14, 5, 0, 4, 4, 0
_memo = {"frame": None, "n": 0}


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def socket_kind(sockets, c):
    for s in sockets:
        x, y, w, h = s
        if x <= c[0] < x + w and y <= c[1] < y + h:
            inner = x < c[0] < x + w - 1 and y < c[1] < y + h - 1
            return "interior" if inner else "border"
    return None


def estimate_terrain(frame, covered, sockets, c):
    kind = socket_kind(sockets, c)
    if kind == "border":
        return 4
    if kind == "interior":
        return 1
    votes = {}
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        x, y = c
        while 0 <= x < 64 and 0 <= y < 63:
            x, y = x + dx, y + dy
            if not (0 <= x < 64 and 0 <= y < 63):
                break
            if (x, y) not in covered and frame[y][x] in TERRAIN:
                votes[frame[y][x]] = votes.get(frame[y][x], 0) + 1
                break
    return max(votes, key=lambda k: votes[k]) if votes else 1


def in_bound(r, bound):
    return bound[0] <= r[0] and bound[1] <= r[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def player_blocked(r, terrain, bound):
    return not in_bound(r, bound) or any(terrain[j][i] not in PLAYER_FLOOR for i, j in cells(r))


def socket_rejects(r, sockets):
    for s in sockets:
        inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, terrain, bound, player, sockets):
    if not in_bound(r, bound) or overlap(r, player) or socket_rejects(r, sockets):
        return True
    return any(terrain[j][i] not in BLOCK_FLOOR for i, j in cells(r))


def push_chain(blocks, first, d, terrain, bound, player, sockets):
    blocks = list(blocks)
    moved = False
    chain = set(first)
    while True:
        changed = True
        while changed:
            changed = False
            for k in list(chain):
                nxt = shift(blocks[k], d)
                for m, b in enumerate(blocks):
                    if m not in chain and overlap(nxt, b):
                        chain.add(m)
                        changed = True
        if any(block_blocked(shift(blocks[k], d), terrain, bound, player, sockets) for k in chain):
            return blocks, moved
        for k in chain:
            blocks[k] = shift(blocks[k], d)
        moved = True


def touching_sides(p, blocks):
    x, y, w, h = p
    sides = set()
    for b in blocks:
        bx, by, bw, bh = b
        xs = bx < x + w and x < bx + bw
        ys = by < y + h and y < by + bh
        if xs and by + bh == y:
            sides.add("top")
        if xs and by == y + h:
            sides.add("bottom")
        if ys and bx + bw == x:
            sides.add("left")
        if ys and bx == x + w:
            sides.add("right")
    return sides


def draw_token(out, r, core):
    x, y, w, h = r
    for i, j in cells(r):
        out[j][i] = RIM
    for j in range((h - 1) // 2, h // 2 + 1):
        for i in range((w - 1) // 2, w // 2 + 1):
            out[y + j][x + i] = core


def draw_player(out, p, blocks):
    draw_token(out, p, PLAYER_CORE)
    x, y, w, h = p
    sides = touching_sides(p, blocks)
    for i in range(x, x + w):
        if "top" in sides:
            out[y][i] = PLAYER_CORE
        if "bottom" in sides:
            out[y + h - 1][i] = PLAYER_CORE
    for j in range(y, y + h):
        if "left" in sides:
            out[j][x] = PLAYER_CORE
        if "right" in sides:
            out[j][x + w - 1] = PLAYER_CORE


def hud_count(frame):
    if _memo["frame"] is not None and frame == _memo["frame"]:
        return _memo["n"]
    return 2 * sum(1 for v in frame[63] if v == HUD_MARK)


def transition_function(state, action, frame):
    player = None
    blocks, sockets = [], []
    bound = (0, 0, 64, 63)
    for o in state:
        if o["type"] == "player":
            player = rect(o)
        elif o["type"] == "block":
            blocks.append(rect(o))
        elif o["type"] == "target":
            sockets.append(rect(o))
        elif o["type"] == "portal":
            bound = rect(o)
    n = hud_count(frame)
    movers = blocks + ([player] if player else [])
    covered = set(c for r in movers for c in cells(r))
    terrain = [row[:] for row in frame]
    for c in covered:
        terrain[c[1]][c[0]] = estimate_terrain(frame, covered, sockets, c)
    new_blocks = list(blocks)
    new_player = player
    if player is not None and isinstance(action, int) and action in DIRS:
        d = DIRS[action]
        step = 3
        nxt = shift(player, d, step)
        hit = [k for k, b in enumerate(blocks) if overlap(nxt, b)]
        if hit:
            new_blocks, _ = push_chain(blocks, hit, d, terrain, bound, player, sockets)
        elif not player_blocked(nxt, terrain, bound):
            new_player = nxt
    elif isinstance(action, dict) and action.get("action_id") == 6 and player is not None:
        c = (action["x"], action["y"])
        for k, b in enumerate(blocks):
            if overlap((c[0], c[1], 1, 1), b):
                draw_token(terrain, player, HUSK_CORE)
                new_player = b
                new_blocks = blocks[:k] + blocks[k + 1:]
                break
    out = [row[:] for row in terrain]
    for b in new_blocks:
        draw_token(out, b, BLOCK_CORE)
    if new_player is not None:
        draw_player(out, new_player, new_blocks)
    n += 1
    zeros = (n + 1) // 2
    out[63] = [HUD_BG] * 64
    for i in range(min(zeros, 64)):
        out[63][63 - i] = HUD_MARK
    _memo["frame"] = [row[:] for row in out]
    _memo["n"] = n
    return out
