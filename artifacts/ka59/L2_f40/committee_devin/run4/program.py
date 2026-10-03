# Mechanics: sokoban-style token game on a 3-px grid. Player moves 3 px per action (A1 up, A2 down, A3 left, A4 right);
# a move into tokens pushes them: the hit chain SLIDES (picking up tokens it touches) until any member would hit a wall,
# while the player stays put. A6 click on a token: the player jumps into it (takes its box), the token is consumed.
# Blocks are renamed token_i by (y, x). Walls have no pixels in the extractor, so the few wall cells are inferred from
# blocked moves/slide stops (unconfirmed elsewhere). 'Mixed A3 outcomes touching portal' = whether the player pushes; no hidden state.
import copy

DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
SIZE = 63
# invisible level walls (x, y, w, h), inferred from where moves/slides stop
WALLS = [(27, 54, 3, 3),     # player blocked moving left from x=30 on row 54
         (33, 27, 12, 3),    # upward slides stop at y=30 in columns 33..44
         (12, 48, 3, 3)]     # 6x6 token sliding left on rows 45..50 stops at x=15


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def box(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(o, dx, dy):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def blocked_by_wall(b):
    if b[0] < 0 or b[1] < 0 or b[0] + b[2] > SIZE or b[1] + b[3] > SIZE:
        return True
    return any(overlap(b, w) for w in WALLS)


def push_group(first, tokens, dx, dy):
    group, frontier = [first], [first]
    while frontier:
        t = frontier.pop()
        nb = shifted(t, dx, dy)
        for u in tokens:
            if all(u is not g for g in group) and overlap(nb, box(u)):
                group.append(u)
                frontier.append(u)
    return group


def slide(first, tokens, dx, dy):
    moved = False
    while True:
        group = push_group(first, tokens, dx, dy)
        if any(blocked_by_wall(shifted(t, dx, dy)) for t in group):
            return moved
        for t in group:
            t["x"] += dx
            t["y"] += dy
        moved = True


def move_player(player, tokens, dx, dy):
    nb = shifted(player, dx, dy)
    if blocked_by_wall(nb):
        return
    hit = [t for t in tokens if overlap(nb, box(t))]
    if not hit:
        player["x"], player["y"] = nb[0], nb[1]
        return
    for t in hit:
        slide(t, tokens, dx, dy)


def click(player, tokens, x, y):
    for t in tokens:
        if overlap((x, y, 1, 1), box(t)):
            player["x"], player["y"], player["w"], player["h"] = t["x"], t["y"], t["w"], t["h"]
            tokens.remove(t)
            return


def rename(tokens):
    for i, t in enumerate(sorted(tokens, key=lambda o: (o["y"], o["x"]))):
        t["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    tokens = [o for o in state if o["type"] == "block"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if not players:
        return state
    player = players[0]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(player, tokens, action["x"], action["y"])
    elif action in DIRS:
        move_player(player, tokens, *DIRS[action])
    rename(tokens)
    return players + tokens + others
