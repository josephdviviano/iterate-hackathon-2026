# Mechanics: player (5x5) moves +-5 per A1-4 (A1 up, A2 down, A3 left, A4 right); invisible maze walls
# (inferred from blocked moves) stop it; a portal cell (9,5) teleports to (34,5). step_bar loses 2 per
# action (x = 55 - w, tag str(w)); a refuel ring fully covered by the player is consumed and refills the
# bar to 42. A target fully covered by the player is occluded (not extracted) and reappears once uncovered
# (remembered via continuity-gated hidden state). Unconfirmed: maze outside observed cells, bounds, bar at 0.
import copy
import json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BOUNDS = (0, 59)
BAR_FULL, BAR_COST, BAR_RIGHT = 42, 2, 55

_last = {"out": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"] and
            o["x"] + o["w"] <= p["x"] + p["w"] and o["y"] + o["h"] <= p["y"] + p["h"])


def blocked(x, y):
    return (x, y) in WALLS or not (BOUNDS[0] <= x <= BOUNDS[1] and BOUNDS[0] <= y <= BOUNDS[1])


def move_player(p, action):
    if action not in MOVES:
        return
    dx, dy = MOVES[action]
    nx, ny = p["x"] + dx, p["y"] + dy
    if blocked(nx, ny):
        return
    p["x"], p["y"] = PORTALS.get((nx, ny), (nx, ny))


def set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t for t in bar["tags"] if not t.isdigit()] + [str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = []
    if _last["out"] is not None and _canon(state) == _last["out"]:
        hidden = copy.deepcopy(_last["hidden"])
    aid = action.get("action_id") if isinstance(action, dict) else action

    players = [o for o in state if o["type"] == "player"]
    bars = [o for o in state if o["type"] == "counter" and "budget" in o.get("tags", [])]
    for p in players:
        move_player(p, aid)

    refuelled = False
    out = []
    for o in state + hidden:
        if o["type"] == "refuel" and any(covers(p, o) for p in players):
            refuelled = True
            continue
        out.append(o)

    for bar in bars:
        set_bar(bar, BAR_FULL if refuelled else bar["w"] - BAR_COST)

    visible, new_hidden = [], []
    for o in out:
        if o["type"] == "target" and "collect" in o.get("tags", []) and any(covers(p, o) for p in players):
            new_hidden.append(o)
        else:
            visible.append(o)

    _last["out"] = _canon(visible)
    _last["hidden"] = copy.deepcopy(new_hidden)
    return visible
