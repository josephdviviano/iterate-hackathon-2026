# Mechanics: arrows (A1 up, A2 down, A3 left, A4 right) move the player 3 cells unless the
# destination leaves the portal bbox or touches frame colour 2 (solid) or 15 (player-only wall).
# A destination overlapping a block kicks it instead (player stays): the push chain slides 1 cell per
# tick, absorbing blocks it touches, until a member hits colour 2, the bound, the player, or a socket
# interior it does not fit. Click on a block: player takes its rect, block removed, tokens re-rank (y,x).
# Unconfirmed: fitting-socket fill behaviour, A5/A7 (treated as no-ops), click on non-blocks (no-op).
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
SOLID = 2
PLAYER_WALL = 15


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.player = next((o for o in state if o["type"] == "player"), None)
        self.blocks = [o for o in state if o["type"] == "block"]
        self.targets = [o for o in state if o["type"] == "target"]
        portal = next((o for o in state if o["type"] == "portal"), None)
        self.bound = rect(portal) if portal else (0, 0, 64, 63)
        self.covered = set()
        for o in [self.player] + self.blocks:
            if o:
                self.covered.update(cells(rect(o)))

    def in_bound(self, r):
        b = self.bound
        return b[0] <= r[0] and b[1] <= r[1] and r[0] + r[2] <= b[0] + b[2] and r[1] + r[3] <= b[1] + b[3]

    def hits_colour(self, r, colours):
        for x, y in cells(r):
            if (x, y) in self.covered:
                continue
            if 0 <= y < len(self.frame) and 0 <= x < len(self.frame[0]) and self.frame[y][x] in colours:
                return True
        return False

    def blocked_by_socket(self, r):
        for t in self.targets:
            inner = (t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2)
            if inner[2] > 0 and inner[3] > 0 and overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
                return True
        return False

    def block_blocked(self, r):
        if not self.in_bound(r) or self.hits_colour(r, (SOLID,)):
            return True
        if self.player and overlap(r, rect(self.player)):
            return True
        return self.blocked_by_socket(r)

    def push_chain(self, seeds, dx, dy):
        chain = list(seeds)
        while True:
            grew = True
            while grew:
                grew = False
                moved = [shift(rect(b), dx, dy) for b in chain]
                for b in self.blocks:
                    if b not in chain and any(overlap(m, rect(b)) for m in moved):
                        chain.append(b)
                        grew = True
            if any(self.block_blocked(shift(rect(b), dx, dy)) for b in chain):
                return
            for b in chain:
                b["x"] += dx
                b["y"] += dy

    def move(self, action):
        if not self.player:
            return
        dx, dy = DIRS[action]
        dest = shift(rect(self.player), dx * STEP, dy * STEP)
        hit = [b for b in self.blocks if overlap(dest, rect(b))]
        if hit:
            self.push_chain(hit, dx, dy)
            return
        if not self.in_bound(dest) or self.hits_colour(dest, (SOLID, PLAYER_WALL)):
            return
        self.player["x"], self.player["y"] = dest[0], dest[1]

    def click(self, x, y):
        b = next((b for b in self.blocks if overlap((x, y, 1, 1), rect(b))), None)
        if b is None or self.player is None:
            return
        for k in ("x", "y", "w", "h"):
            self.player[k] = b[k]
        self.blocks.remove(b)

    def output(self, state):
        order = sorted(self.blocks, key=lambda b: (b["y"], b["x"]))
        for i, b in enumerate(order):
            b["name"] = "token_%d" % i
        keep = [o for o in state if o["type"] not in ("player", "block")]
        return ([self.player] if self.player else []) + order + keep


def transition_function(state, action, frame=None):
    state = copy.deepcopy(state)
    w = World(state, frame or [])
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            w.click(action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        w.move(action)
    return w.output(state)
