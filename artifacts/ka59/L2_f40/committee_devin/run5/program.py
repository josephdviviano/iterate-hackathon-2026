# Mechanics: arrows (1 up, 2 down, 3 left, 4 right) move the player 3 cells; if the player's next
# rect overlaps a block, that block is kicked instead (player stays) and slides 3/step, pushing any
# blocks it runs into, until the group hits an invisible wall, the board edge, the player, or the
# interior of a socket it does not fit. Clicking a block makes it the player (old player vanishes).
# Unconfirmed: walls beyond the two inferred rects; a fitting block entering a socket (assumed: stops filling it).
import copy

STEP = 3
BOARD = 63
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = [(27, 54, 3, 3), (33, 27, 12, 3)]  # inferred from blocked moves (x, y, w, h)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def in_bounds(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= BOARD and r[1] + r[3] <= BOARD


def interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def fits(r, s):
    i = interior(s)
    return (r[2], r[3]) == (i[2], i[3])


def wall_hit(r):
    return not in_bounds(r) or any(overlaps(r, w) for w in WALLS)


def block_blocked(r, sockets, player_r):
    if wall_hit(r) or overlaps(r, player_r):
        return True
    return any(overlaps(r, interior(s)) and not fits(r, s) for s in sockets)


def seated(r, sockets):
    return any(r == interior(s) for s in sockets)


def slide(blocks, start, d, sockets, player_r):
    """blocks: list of rects. Slide block `start` (pushing chains) until blocked."""
    rects = list(blocks)
    while True:
        group, frontier = {start}, [start]
        while frontier:
            i = frontier.pop()
            nr = shift(rects[i], d)
            for j, r in enumerate(rects):
                if j not in group and overlaps(nr, r):
                    group.add(j)
                    frontier.append(j)
        moved = {i: shift(rects[i], d) for i in group}
        if any(block_blocked(r, sockets, player_r) for r in moved.values()):
            return rects
        for i, r in moved.items():
            rects[i] = r
        if any(seated(r, sockets) for r in moved.values()):
            return rects


def relabel(blocks):
    blocks.sort(key=lambda o: (o["y"], o["x"]))
    for k, o in enumerate(blocks):
        o["name"] = "token_%d" % k


def set_rect(o, r):
    o["x"], o["y"], o["w"], o["h"] = r


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state
    pr = rect(player)

    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            hit = [b for b in blocks if overlaps((cx, cy, 1, 1), rect(b))]
            if hit:
                b = hit[0]
                set_rect(player, rect(b))
                blocks.remove(b)
        relabel(blocks)
        return [player] + blocks + others

    d = DIRS.get(action)
    if d is None:
        return state
    nr = shift(pr, d)
    kicked = [i for i, b in enumerate(blocks) if overlaps(nr, rect(b))]
    if kicked:
        rects = slide([rect(b) for b in blocks], kicked[0], d, sockets, pr)
        for b, r in zip(blocks, rects):
            set_rect(b, r)
    elif not wall_hit(nr):
        set_rect(player, nr)
    relabel(blocks)
    return [player] + blocks + others
