# Mechanics: player (5x5) moves 5 cells per action (1 up, 2 down, 3 left, 4 right) on an invisible maze;
# blocked moves are no-ops. Entering the portal cell (9,5) teleports the player to (34,5).
# step_bar (budget counter) loses 2 per action (right-anchored at x=55, tag = width); touching a refuel
# ring consumes it and refills the bar to 42. Targets covered by the player are occluded (not extracted)
# and reappear when uncovered (remembered via continuity-gated hidden state). Unconfirmed: walls inferred only from no-ops.
import copy, json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}          # player cells inferred from blocked moves
PORTALS = {(9, 5): (34, 5)}
BAR_MAX, BAR_RIGHT, BAR_COST = 42, 55, 2

_mem = {"last": None, "occluded": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def overlaps(a, b):
    return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and
            a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])


def update_player(p, action):
    if not isinstance(action, int) or action not in MOVES:
        return
    dx, dy = MOVES[action]
    nxt = (p["x"] + dx, p["y"] + dy)
    if nxt in WALLS:
        return
    p["x"], p["y"] = PORTALS.get(nxt, nxt)


def set_bar(bar, w):
    w = max(0, min(BAR_MAX, w))
    bar["w"], bar["x"] = w, BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    occluded = _mem["occluded"] if _mem["last"] == _canon(state) else []
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    for p in players:
        update_player(p, action)
    out, refuel = [], False
    for o in state + copy.deepcopy(occluded):
        if o["type"] == "refuel" and any(overlaps(p, o) for p in players):
            refuel = True
            continue
        out.append(o)
    new_occ = []
    final = []
    for o in out:
        if o["type"] == "target" and "collect" in o["tags"] and any(overlaps(p, o) for p in players):
            new_occ.append(o)
            continue
        if o["type"] == "counter" and o["name"] == "step_bar":
            set_bar(o, BAR_MAX if refuel else o["w"] - BAR_COST)
        final.append(o)
    _mem["last"], _mem["occluded"] = _canon(final), new_occ
    return final
