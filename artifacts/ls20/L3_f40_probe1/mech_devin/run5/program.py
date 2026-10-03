# Mechanics: player (5x5, layer 1) moves 5 cells on A1-4 (up/down/left/right); invisible maze walls block some cells
# (inferred {(39,5),(34,10)}), entering portal cell (9,5) warps to (34,5). Every move attempt burns 2 from step_bar
# (w-=2, x=55-w, tag=str(w)), even when blocked. Fit guards: player fully covering a refuel ring consumes it and refills
# the bar to 42; fully covering a 'chamber' target rejects the move outright at no cost (state unchanged); fully covering
# another target occludes it (kept in continuity-gated memory, restored when uncovered). Unconfirmed: what unlocks the chamber.
import copy, json

MOVES = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_MAX, BAR_RIGHT, BURN = 42, 55, 2
BOARD = 64

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(p, px, py, o):
    return (px <= o["x"] and py <= o["y"] and o["x"] + o["w"] <= px + p["w"]
            and o["y"] + o["h"] <= py + p["h"])


def _is_chamber(o):
    return o["type"] == "target" and "chamber" in o.get("tags", [])


def _set_bar(bar, w):
    w = max(0, min(BAR_MAX, w))
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def _target_cell(px, py, dx, dy):
    nx, ny = px + dx, py + dy
    if (nx, ny) in WALLS:
        return px, py
    return PORTALS.get((nx, ny), (nx, ny))


def transition_function(state, action):
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    out = copy.deepcopy(state)
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in out if o["type"] == "player"), None)
    bar = next((o for o in out if o["type"] == "counter" and "budget" in o.get("tags", [])), None)
    if player is None or aid not in MOVES:
        _mem["last"], _mem["hidden"] = _canon(out), hidden
        return out
    dx, dy = MOVES[aid]
    nx, ny = _target_cell(player["x"], player["y"], dx, dy)
    off_board = nx < 0 or ny < 0 or nx + player["w"] > BOARD or ny + player["h"] > BOARD
    if off_board or any(_is_chamber(o) and _covers(player, nx, ny, o) for o in out):
        _mem["last"], _mem["hidden"] = _canon(out), hidden
        return out
    player["x"], player["y"] = nx, ny
    if bar is not None:
        _set_bar(bar, bar["w"] - BURN)
    keep, new_hidden = [], []
    for o in out:
        if o is player or o is bar or o["layer"] >= player["layer"] or not _covers(player, nx, ny, o):
            keep.append(o)
        elif o["type"] == "refuel":
            if bar is not None:
                _set_bar(bar, BAR_MAX)
        elif not _is_chamber(o):
            new_hidden.append(o)
    for o in hidden:
        (new_hidden if _covers(player, nx, ny, o) else keep).append(o)
    _mem["last"], _mem["hidden"] = _canon(keep), new_hidden
    return keep
