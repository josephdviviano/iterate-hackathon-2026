# Kick/slide token game. A1-A4 step the player 3px; a step off the floor rects is a
# no-op, while a step into a pushable token kicks it: the struck token plus tokens
# caught in its swept path slide in that direction until stopped by the floor edge,
# a socket interior they do not exactly match (w,h), or another token/player.
# action_id 6 clicks: the player possesses the clicked token (takes its x,y,w,h;
# token removed). Tokens are re-extracted as token_i ranked by (y,x) each frame.
# Unconfirmed: A5/A7 (unseen -> no-op), socket 'filled' tagging on an exact-fit dock.

import copy

# Inclusive pixel rects walkable by tokens and the player (terrain is decorative).
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _on_floor(x, y, w, h):
    for py in range(y, y + h):
        for px in range(x, x + w):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def _interiors(sockets):
    for s in sockets:
        yield (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def _socket_blocks(x, y, w, h, sockets):
    # A token may not overlap a socket's inset-1 interior unless its w,h match
    # that interior exactly (a docking fit).
    for ix, iy, iw, ih in _interiors(sockets):
        if iw <= 0 or ih <= 0:
            continue
        if _overlap(x, y, w, h, ix, iy, iw, ih):
            if not (w == iw and h == ih):
                return True
    return False


def _block_ok(x, y, w, h, sockets, obstacles):
    if not _on_floor(x, y, w, h):
        return False
    if _socket_blocks(x, y, w, h, sockets):
        return False
    for ox, oy, ow, oh in obstacles:
        if _overlap(x, y, w, h, ox, oy, ow, oh):
            return False
    return True


def _travel_limit(b, dx, dy, sockets):
    # Farthest position if only floor and socket interiors constrained the slide.
    x, y = b["x"], b["y"]
    while _on_floor(x + dx, y + dy, b["w"], b["h"]) and not _socket_blocks(
        x + dx, y + dy, b["w"], b["h"], sockets
    ):
        x += dx
        y += dy
    return x, y


def _push_chain(blocks, hit, dx, dy, sockets):
    # Closure: any token overlapped by a chained token's potential sweep joins.
    chain = set(hit)
    grew = True
    while grew:
        grew = False
        for i in list(chain):
            b = blocks[i]
            ex, ey = _travel_limit(b, dx, dy, sockets)
            sx0, sx1 = min(b["x"], ex), max(b["x"] + b["w"], ex + b["w"])
            sy0, sy1 = min(b["y"], ey), max(b["y"] + b["h"], ey + b["h"])
            for j, ob in enumerate(blocks):
                if j in chain:
                    continue
                if _overlap(sx0, sy0, sx1 - sx0, sy1 - sy0,
                            ob["x"], ob["y"], ob["w"], ob["h"]):
                    chain.add(j)
                    grew = True
    return chain


def _slide_chain(blocks, chain, dx, dy, sockets, static_obstacles):
    pos = {i: (blocks[i]["x"], blocks[i]["y"]) for i in chain}
    key = (lambda i: pos[i][0]) if dx < 0 else \
          (lambda i: -pos[i][0]) if dx > 0 else \
          (lambda i: pos[i][1]) if dy < 0 else (lambda i: -pos[i][1])
    moved = {}
    for i in sorted(chain, key=key):
        b = blocks[i]
        x, y = pos[i]
        others = []
        for j, ob in enumerate(blocks):
            if j == i:
                continue
            others.append(moved.get(j, (ob["x"], ob["y"])) + (ob["w"], ob["h"]))
        others += static_obstacles
        while _block_ok(x + dx, y + dy, b["w"], b["h"], sockets, others):
            x += dx
            y += dy
        moved[i] = (x, y)
    return moved


def _rename_tokens(state):
    blocks = sorted((o for o in state if o.get("type") == "block"),
                    key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(blocks):
        o["name"] = "token_%d" % i


def transition_function(state, action):
    out = copy.deepcopy(state)
    player = next((o for o in out if o.get("type") == "player"), None)
    blocks = [o for o in out if o.get("type") == "block"]
    sockets = [o for o in out if "socket" in (o.get("tags") or [])]

    if isinstance(action, dict):
        if action.get("action_id") == 6 and player is not None:
            cx, cy = action.get("x"), action.get("y")
            hit = next(
                (b for b in blocks
                 if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]),
                None)
            if hit is not None:
                player["x"], player["y"] = hit["x"], hit["y"]
                player["w"], player["h"] = hit["w"], hit["h"]
                out.remove(hit)
                _rename_tokens(out)
        return out

    if player is None or action not in DIRS:
        return out
    dx, dy = DIRS[action]
    step = 3
    nx, ny = player["x"] + dx * step, player["y"] + dy * step

    hit = [i for i, b in enumerate(blocks)
           if _overlap(nx, ny, player["w"], player["h"],
                       b["x"], b["y"], b["w"], b["h"])]
    if hit:
        chain = _push_chain(blocks, hit, dx, dy, sockets)
        static = [(player["x"], player["y"], player["w"], player["h"])]
        moved = _slide_chain(blocks, chain, dx, dy, sockets, static)
        if all(moved[i] == (blocks[i]["x"], blocks[i]["y"]) for i in chain):
            return out
        for i, (bx, by) in moved.items():
            blocks[i]["x"], blocks[i]["y"] = bx, by
        _rename_tokens(out)
        return out

    if _on_floor(nx, ny, player["w"], player["h"]):
        player["x"], player["y"] = nx, ny
    return out
