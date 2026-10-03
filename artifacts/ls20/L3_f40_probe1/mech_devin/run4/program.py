# Mechanics: player (5x5) steps 5 cells per A1-4 (up/down/left/right); invisible walls block cells
# {(39,5),(34,10)} (blocked move still costs); portal: landing on (9,5) warps to (34,5).
# Every action burns 2 from step_bar (w-=2, x=55-w, tag=str(w)); a refuel ring fully covered by the
# player is consumed and refills the bar to 42. Layer-0 targets fully covered are occluded (hidden
# memory, continuity-gated) and reappear when uncovered. Move into a chamber/off-board = free no-op.
import copy, json

WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BOARD = 64
FULL_BAR, BAR_RIGHT, BURN = 42, 55, 2
DELTA = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}

_mem = {"last": None, "hidden": []}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"] and
            o["x"] + o["w"] <= p["x"] + p["w"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _overlaps(ax, ay, aw, ah, o):
    return ax < o["x"] + o["w"] and o["x"] < ax + aw and ay < o["y"] + o["h"] and o["y"] < ay + ah


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def _chamber_blocked(state, nx, ny, p):
    if nx < 0 or ny < 0 or nx + p["w"] > BOARD or ny + p["h"] > BOARD:
        return True
    return any("chamber" in o.get("tags", []) and _overlaps(nx, ny, p["w"], p["h"], o) for o in state)


def transition_function(state, action):
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    state = copy.deepcopy(state)
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in state if o["type"] == "player"), None)
    bar = next((o for o in state if o["name"] == "step_bar"), None)

    if player is not None and aid in DELTA:
        dx, dy = DELTA[aid]
        nx, ny = player["x"] + dx, player["y"] + dy
        if _chamber_blocked(state, nx, ny, player):
            _mem["last"], _mem["hidden"] = _canon(state), hidden
            return state
        if (nx, ny) not in WALLS:
            nx, ny = PORTALS.get((nx, ny), (nx, ny))
            player["x"], player["y"] = nx, ny

    if bar is not None:
        _set_bar(bar, bar["w"] - BURN)

    out = []
    for o in state:
        if player is not None and o is not player and o["type"] == "refuel" and _covers(player, o):
            if bar is not None:
                _set_bar(bar, FULL_BAR)
            continue
        out.append(o)

    new_hidden = []
    for o in hidden:
        if player is not None and _covers(player, o):
            new_hidden.append(o)
        else:
            out.append(o)
    final = []
    for o in out:
        if (player is not None and o is not player and o["type"] == "target" and o["layer"] < player["layer"]
                and _covers(player, o)):
            new_hidden.append(o)
        else:
            final.append(o)
    _mem["last"], _mem["hidden"] = _canon(final), new_hidden
    return final
