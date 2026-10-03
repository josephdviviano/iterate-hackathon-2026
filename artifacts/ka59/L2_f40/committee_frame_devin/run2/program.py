# Mechanics: arrows move the player 3 cells; terrain is read from the frame: colour 2 (and colour 4 outside a
# socket bbox, or off-board) is wall for everything, colour 15 is wall for the player only (blocks slide over it).
# If the player's next rect overlaps a token it is kicked instead (player stays): the token slides 3/step, picking up
# every token it hits (chain), until any member hits a wall, the player, or a socket interior it does not fit.
# Click on a token: the player takes its rect, token removed; tokens re-ranked token_i by (y,x). Unconfirmed: docking.
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
WALL_ALL = 2
WALL_PLAYER_ONLY = 15
SOCKET_BORDER = 4


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def moved(r, d):
    return (r[0] + d[0] * STEP, r[1] + d[1] * STEP, r[2], r[3])


def inside(r, x, y):
    return r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.sockets = [rect(o) for o in state if o["type"] == "target"]

    def cell_blocks(self, x, y, is_player):
        if not (0 <= y < len(self.frame) and 0 <= x < len(self.frame[0])):
            return True
        c = self.frame[y][x]
        if c == WALL_ALL:
            return True
        if c == SOCKET_BORDER and not any(inside(s, x, y) for s in self.sockets):
            return True
        return is_player and c == WALL_PLAYER_ONLY

    def terrain_blocks(self, r, is_player):
        return any(self.cell_blocks(x, y, is_player)
                   for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2]))

    def socket_rejects(self, r):
        for s in self.sockets:
            interior = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
            if overlaps(r, interior) and (r[2], r[3]) != (interior[2], interior[3]):
                return True
        return False

    def block_can_enter(self, r):
        return not self.terrain_blocks(r, False) and not self.socket_rejects(r)


def slide_chain(world, blocks, player_r, first, d):
    """Slide the kicked chain step by step; returns new rects for blocks."""
    rects = [rect(b) for b in blocks]
    group = {first}
    moved_any = False
    while True:
        changed = True
        while changed:
            changed = False
            for i in list(group):
                nr = moved(rects[i], d)
                for j, r in enumerate(rects):
                    if j not in group and overlaps(nr, r):
                        group.add(j)
                        changed = True
        if any(not world.block_can_enter(moved(rects[i], d)) or overlaps(moved(rects[i], d), player_r)
               for i in group):
            break
        for i in group:
            rects[i] = moved(rects[i], d)
        moved_any = True
    return rects, moved_any


def rename_tokens(blocks):
    blocks.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


def transition_function(state, action, frame=None):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    blocks = [o for o in state if o["type"] == "block"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if not players or frame is None:
        return state
    player = players[0]
    world = World(state, frame)

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if inside(rect(b), cx, cy)]
        if hit:
            b = hit[0]
            player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
            blocks.remove(b)
    elif action in DIRS:
        d = DIRS[action]
        pr = rect(player)
        nr = moved(pr, d)
        hit = [i for i, b in enumerate(blocks) if overlaps(nr, rect(b))]
        if hit:
            rects, ok = slide_chain(world, blocks, pr, hit[0], d)
            if ok:
                for b, r in zip(blocks, rects):
                    b["x"], b["y"] = r[0], r[1]
        elif not world.terrain_blocks(nr, True):
            player["x"], player["y"] = nr[0], nr[1]

    rename_tokens(blocks)
    return [player] + blocks + others
