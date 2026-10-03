# Mechanics: player (5x5 block) moves 5 cells per ACTION1-4; invisible walls {(39,5),(34,10)} block it,
# entering portal cell (9,5) warps to (34,5). Every action burns 2 from step_bar (x = 55 - w, tag = str(w)),
# except a move into a 'chamber' target (legend) or off the board: refused entirely, nothing changes.
# A refuel ring fully covered by the player is consumed and refills step_bar to 42; other layer-0 objects
# fully covered are occluded (not extracted) and reappear when uncovered (continuity-gated memory).
# Unconfirmed: what unlocks the chamber; walls beyond the two observed cells; actions 5-7 (treated as burn-only).
import copy
import json

STEP = 5
BOARD = 64
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
BAR_MAX, BAR_COST, BAR_RIGHT = 42, 2, 55

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(box, o):
    x, y, w, h = box
    return x <= o["x"] and y <= o["y"] and o["x"] + o["w"] <= x + w and o["y"] + o["h"] <= y + h


def _overlaps(box, o):
    x, y, w, h = box
    return x < o["x"] + o["w"] and o["x"] < x + w and y < o["y"] + o["h"] and o["y"] < y + h


def _is_chamber(o):
    return o["type"] == "target" and "chamber" in o.get("tags", [])


def _player_dest(p, action):
    dx, dy = DIRS[action]
    nx, ny = p["x"] + dx * STEP, p["y"] + dy * STEP
    if (nx, ny) in WALLS:
        return p["x"], p["y"]
    return PORTALS.get((nx, ny), (nx, ny))


def _refused(state, p, dest):
    nx, ny = dest
    if nx < 0 or ny < 0 or nx + p["w"] > BOARD or ny + p["h"] > BOARD:
        return True
    box = (nx, ny, p["w"], p["h"])
    return any(_is_chamber(o) and _overlaps(box, o) for o in state)


def _burn(bar, amount=BAR_COST):
    w = max(bar["w"] - amount, 0)
    _set_bar(bar, w)


def _set_bar(bar, w):
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    state = copy.deepcopy(state)
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in state if o["type"] == "player"), None)
    bar = next((o for o in state if o["name"] == "step_bar"), None)

    if player is not None and aid in DIRS:
        dest = _player_dest(player, aid)
        if _refused(state, player, dest):
            _mem["last"], _mem["hidden"] = _canon(state), hidden
            return state
        player["x"], player["y"] = dest
    if bar is not None:
        _burn(bar)

    out = [o for o in state]
    if player is not None:
        box = (player["x"], player["y"], player["w"], player["h"])
        # previously hidden objects reappear when uncovered
        still_hidden = []
        for o in hidden:
            (still_hidden if _covers(box, o) else out).append(o)
        kept = []
        for o in out:
            if o is player or o.get("layer", 0) != 0 or not _covers(box, o):
                kept.append(o)
            elif o["type"] == "refuel":
                if bar is not None:
                    _set_bar(bar, BAR_MAX)
            else:
                still_hidden.append(o)
        out, hidden = kept, still_hidden
    _mem["last"], _mem["hidden"] = _canon(out), hidden
    return out
