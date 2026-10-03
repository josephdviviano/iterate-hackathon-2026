# Mechanics: player (5x5) moves 5 cells per arrow (1 up,2 down,3 left,4 right); hidden walls at
# player cells (39,5),(34,10) block the move (still costs a step); portal cell (9,5) warps to (34,5).
# step_bar counter burns 2 per step (tag=str(w), x=55-w); a refuel ring under the player is consumed
# and refills the bar to 42. Layer-0 targets under the player are occluded (remembered, continuity-gated).
# Guard: a move off-board or onto a 'chamber' target is refused entirely (no cost). Unconfirmed: what unlocks it.
import copy, json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BOARD = 64
BAR_FULL, BAR_RIGHT, BURN = 42, 55, 2

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _overlap(a, b):
    return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
            and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])


def _box(x, y, p):
    return {"x": x, "y": y, "w": p["w"], "h": p["h"]}


def refused(box, state):
    off = box["x"] < 0 or box["y"] < 0 or box["x"] + box["w"] > BOARD or box["y"] + box["h"] > BOARD
    chamber = any("chamber" in o.get("tags", []) and _overlap(box, o) for o in state)
    return off or chamber


def update_player(p, action, state):
    """Return new (x, y) or None if the move is refused."""
    dx, dy = MOVES[action]
    nx, ny = p["x"] + dx, p["y"] + dy
    if refused(_box(nx, ny, p), state):
        return None
    if (nx, ny) in WALLS:
        return p["x"], p["y"]
    return PORTALS.get((nx, ny), (nx, ny))


def set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = _mem["hidden"] if _mem["last"] == _canon(state) else []
    out = _step(state, action, copy.deepcopy(hidden))
    _mem["last"] = _canon(out)
    return out


def _step(state, action, hidden):
    if not isinstance(action, int) or action not in MOVES:
        _mem["hidden"] = hidden
        return state
    player = next((o for o in state if o["type"] == "player"), None)
    if player is None:
        _mem["hidden"] = hidden
        return state
    pos = update_player(player, action, state)
    if pos is None:
        _mem["hidden"] = hidden
        return state
    player["x"], player["y"] = pos
    pbox = _box(pos[0], pos[1], player)
    bar = next((o for o in state if o["type"] == "counter" and "budget" in o.get("tags", [])), None)
    if bar is not None:
        set_bar(bar, bar["w"] - BURN)
    objs = state + hidden
    out, new_hidden = [], []
    for o in objs:
        if o is player or o["type"] == "counter" or o.get("layer", 0) != 0 or not _overlap(pbox, o):
            out.append(o)
        elif o["type"] == "refuel":
            if bar is not None:
                set_bar(bar, BAR_FULL)
        else:
            new_hidden.append(o)
    _mem["hidden"] = new_hidden
    return out
