# Mechanics: player (5x5, layer 1) moves 5 cells per A1-4 (up/down/left/right); hidden maze walls block (39,5),(34,10); portal (9,5)->(34,5).
# step_bar counter: every accepted action costs 2 (w-=2, x=55-w, tag=str(w)), including wall-blocked no-op moves.
# Refuel ring fully covered by the player is consumed and refills the bar to 42. Covered targets are occluded (not extracted)
# and reappear when uncovered (continuity-gated memory). Chamber guard: a move whose destination overlaps a 'chamber' target
# (legend, HUD exit) or leaves the board is refused outright: state unchanged, no cost. Unconfirmed: what unlocks the chamber.
import copy, json

STEP = 5
BAR_RIGHT, BAR_MAX, BAR_COST = 55, 42, 2
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BOARD = 64
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

_mem = {"last": None, "hidden": []}


def _key(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(a, b):
    return (a["x"] <= b["x"] and a["y"] <= b["y"] and
            b["x"] + b["w"] <= a["x"] + a["w"] and b["y"] + b["h"] <= a["y"] + a["h"])


def _overlaps(a, b):
    return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and
            a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])


def _is_chamber(o):
    return o["type"] == "target" and "chamber" in o.get("tags", [])


def _move_player(p, action, objs):
    """Returns new (x, y), or None if the move is refused outright."""
    if action not in DIRS:
        return (p["x"], p["y"])
    dx, dy = DIRS[action]
    nx, ny = p["x"] + dx * STEP, p["y"] + dy * STEP
    dest = dict(p, x=nx, y=ny)
    if nx < 0 or ny < 0 or nx + p["w"] > BOARD or ny + p["h"] > BOARD:
        return None
    if any(_is_chamber(o) and _overlaps(dest, o) for o in objs):
        return None
    if (nx, ny) in WALLS:
        return (p["x"], p["y"])
    return PORTALS.get((nx, ny), (nx, ny))


def _update_bar(bar, refuel):
    w = BAR_MAX if refuel else max(0, bar["w"] - BAR_COST)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t if not t.isdigit() else str(w) for t in bar["tags"]]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = _mem["hidden"] if _mem["last"] is not None and _key(state) == _mem["last"] else []
    if isinstance(action, dict):
        action = action.get("action_id")
    objs = state + [copy.deepcopy(h) for h in hidden]
    player = next((o for o in objs if o["type"] == "player"), None)
    bar = next((o for o in objs if o["type"] == "counter" and "budget" in o.get("tags", [])), None)
    if player is None:
        return state
    pos = _move_player(player, action, objs)
    if pos is None:
        _mem["last"], _mem["hidden"] = _key(state), hidden
        return state
    player["x"], player["y"] = pos
    out, new_hidden, refuel = [], [], False
    for o in objs:
        if o is not player and o["layer"] < player["layer"] and _covers(player, o):
            if o["type"] == "refuel":
                refuel = True
                continue
            if o["type"] == "target":
                new_hidden.append(o)
                continue
        out.append(o)
    if bar is not None:
        _update_bar(bar, refuel)
    _mem["last"], _mem["hidden"] = _key(out), new_hidden
    return out
