# Mechanics: 3px lattice; act1/2=y-,y+ and act3/4=x-,x+ move the player inside a
# floor (room x30-59,y30-59 + corridor x6-29,y42-50; terrain pixels are decor).
# Moving into a pushable block instead slides it (player stays); a sliding block
# pushes any block it contacts, the chain packs until floor edge / socket
# interior (inset 1) blocks it. Click(6) promotes a block's rect to the player.
# Tokens re-rank token_i by (y,x). Unconfirmed: acts 2/5/7, fitting sockets.

FLOOR = ((30, 30, 59, 59), (6, 42, 29, 50))  # inclusive (x0,y0,x1,y1) rects


def _cells(x, y, w, h):
    return set((x + dx, y + dy) for dx in range(w) for dy in range(h))


def _on_floor(x, y, w, h):
    """All bbox cells inside some floor rectangle."""
    for dx in range(w):
        for dy in range(h):
            cx, cy = x + dx, y + dy
            if not any(a <= cx <= c and b <= cy <= d
                       for a, b, c, d in FLOOR):
                return False
    return True


def _socket_interiors(sockets):
    cells = set()
    for s in sockets:
        cells |= _cells(s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
    return cells


def _slide_chain(blocks, seed, dx, dy, hard):
    """Slide pushed block(s) one cell at a time; contacted blocks join in."""
    chain = set(map(id, seed))
    for _ in range(40):
        moved = False
        for b in blocks:
            if id(b) not in chain:
                continue
            while True:
                nx, ny = b["x"] + dx, b["y"] + dy
                if not _on_floor(nx, ny, b["w"], b["h"]):
                    break
                if _cells(nx, ny, b["w"], b["h"]) & hard:
                    break
                hit = next((o for o in blocks if o is not b and _cells(
                    nx, ny, b["w"], b["h"]) & _cells(
                        o["x"], o["y"], o["w"], o["h"])), None)
                if hit is not None:
                    chain.add(id(hit))
                    break
                b["x"], b["y"] = nx, ny
                moved = True
        if not moved:
            break


def transition_function(state, action):
    state = [dict(o) for o in state]
    player = next(o for o in state if o["type"] == "player")
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    block_hard = _socket_interiors(sockets)

    if isinstance(action, dict):  # click: clicked block's rect becomes player
        cx, cy = action.get("x", -1), action.get("y", -1)
        hit = next((b for b in blocks
                    if b["x"] <= cx < b["x"] + b["w"]
                    and b["y"] <= cy < b["y"] + b["h"]), None)
        if hit is not None:
            for k in ("x", "y", "w", "h"):
                player[k] = hit[k]
            state.remove(hit)
            blocks.remove(hit)
    else:
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action, (0, 0))
        if (dx, dy) != (0, 0):
            nx, ny = player["x"] + 3 * dx, player["y"] + 3 * dy
            dest = _cells(nx, ny, player["w"], player["h"])
            pushed = [b for b in blocks if _cells(
                b["x"], b["y"], b["w"], b["h"]) & dest]
            if pushed:
                _slide_chain(blocks, pushed, dx, dy, block_hard)
            elif _on_floor(nx, ny, player["w"], player["h"]):
                player["x"], player["y"] = nx, ny

    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i
    return state
