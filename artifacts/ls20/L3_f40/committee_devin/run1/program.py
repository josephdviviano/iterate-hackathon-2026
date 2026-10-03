# Mechanics: the 5x5 player steps one 5px cell per action (1:y-5, 2:y+5,
# 3:x-5, 4:x+5) along a fixed corridor maze; off-corridor moves are no-ops.
# Cell (9,5) is a portal that lands the player at (34,5). Every action burns
# 2 budget: step_bar w-2/x+2 (right edge fixed at 55); landing on a refuel
# ring consumes it and refills the bar to 42. The layer-1 player occludes
# covered layer-0 objects -> they drop out and return when uncovered.
# Unconfirmed: portal destination seen once; maze only known where walked.

import json, copy

CELL = 5
MOVES = {1: (0, -CELL), 2: (0, CELL), 3: (-CELL, 0), 4: (CELL, 0)}
PORTALS = {(9, 5): (34, 5)}
BAR_RIGHT, BAR_FULL, BAR_COST = 55, 42, 2


def _cells(x0, x1, y0, y1):
    return {(x, y) for x in range(x0, x1 + 1, CELL)
            for y in range(y0, y1 + 1, CELL)}


# Walkable cells = the observed corridors of this level's maze.
WALK = set().union(
    _cells(9, 9, 10, 45),    # entry column
    _cells(29, 34, 5, 5),    # dead-end pocket holding the portal exit
    _cells(29, 29, 5, 15),   # column down from the pocket
    _cells(29, 34, 15, 15),  # short row
    _cells(34, 34, 15, 25),  # column down to the long row
    _cells(34, 54, 25, 25),  # long row
    _cells(54, 54, 10, 25),  # column up on the right
    _cells(49, 54, 10, 10),  # short row left
    _cells(49, 49, 5, 10),   # dead-end up (diamond cell at its foot)
)

_MEM = {"last": None, "hidden": {}}


def _canon(objs):
    return sorted(json.dumps(o, sort_keys=True) for o in objs)


def _covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"]
            and o["x"] + o["w"] <= p["x"] + p["w"]
            and o["y"] + o["h"] <= p["y"] + p["h"])


def _refresh_bar(o, refill):
    old_w = o["w"]
    new_w = BAR_FULL if refill else max(0, old_w - BAR_COST)
    o["w"], o["x"] = new_w, BAR_RIGHT - new_w
    o["tags"] = [str(new_w) if t == str(old_w) else t for t in o["tags"]]


def transition_function(state, action):
    if _MEM["last"] is not None and _MEM["last"] != _canon(state):
        _MEM["hidden"] = {}
    present = {o["name"] for o in state}
    _MEM["hidden"] = {n: o for n, o in _MEM["hidden"].items()
                      if n not in present}

    out = [copy.deepcopy(o) for o in state]
    player = next((o for o in out if o.get("type") == "player"), None)

    moved = False
    if player is not None and isinstance(action, int) and action in MOVES:
        dx, dy = MOVES[action]
        dest = (player["x"] + dx, player["y"] + dy)
        if dest in PORTALS:
            player["x"], player["y"] = PORTALS[dest]
            moved = True
        elif dest in WALK:
            player["x"], player["y"] = dest
            moved = True

    refill = False
    if player is not None and moved:
        kept = []
        for o in out:
            if o is player or not _covers(player, o):
                kept.append(o)
            elif o.get("type") == "refuel":
                refill = True            # consumed for good, budget refilled
            else:
                _MEM["hidden"][o["name"]] = o   # occluded, may return
        out = kept
        for n, o in list(_MEM["hidden"].items()):
            if not _covers(player, o):
                out.append(copy.deepcopy(o))
                del _MEM["hidden"][n]

    for o in out:
        if o.get("type") == "counter" and "budget" in o.get("tags", []):
            _refresh_bar(o, refill)

    _MEM["last"] = _canon(out)
    return out
