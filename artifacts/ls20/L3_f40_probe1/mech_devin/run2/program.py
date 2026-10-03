# Mechanics: player (5x5) moves 5 cells per A1-4 (up/down/left/right); invisible maze walls
# {(39,5),(34,10)} block moves (still cost budget); entering portal cell (9,5) warps to (34,5).
# step_bar burns 2 per action (w-=2, x=55-w, tag=str(w)); a refuel ring fully covered by the player is
# consumed and refills the bar to 42; a layer-0 target fully covered is occluded (memory restores it).
# Guard (global state): while hud_glyph is tagged 'pending', a move into a 'chamber' object is refused
# outright and costs nothing. Unconfirmed: what unlocks the chamber; behaviour at board edges.
import json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_RIGHT = 55
BAR_FULL = 42
BURN = 2

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"] and
            o["x"] + o["w"] <= p["x"] + p["w"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _overlaps(x, y, w, h, o):
    return x < o["x"] + o["w"] and o["x"] < x + w and y < o["y"] + o["h"] and o["y"] < y + h


def _chamber_locked(state):
    return any(o["type"] == "counter" and "pending" in o.get("tags", []) for o in state)


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    state = json.loads(json.dumps(state))
    hidden = _mem["hidden"] if (_mem["last"] is not None and _canon(state) == _mem["last"]) else []
    hidden = json.loads(json.dumps(hidden))
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in state if o["type"] == "player"), None)
    bar = next((o for o in state if o["type"] == "counter" and "budget" in o.get("tags", [])), None)

    if player is not None and aid in MOVES:
        dx, dy = MOVES[aid]
        nx, ny = player["x"] + dx, player["y"] + dy
        if _chamber_locked(state) and any(
                "chamber" in o.get("tags", []) and _overlaps(nx, ny, player["w"], player["h"], o)
                for o in state if o is not player):
            _mem["last"], _mem["hidden"] = _canon(state), hidden
            return state
        if (nx, ny) not in WALLS:
            nx, ny = PORTALS.get((nx, ny), (nx, ny))
            player["x"], player["y"] = nx, ny

    if bar is not None:
        _set_bar(bar, bar["w"] - BURN)

    out = []
    if player is not None:
        for o in state:
            if o is player or o is bar:
                continue
            if o["type"] == "refuel" and _covers(player, o):
                if bar is not None:
                    _set_bar(bar, BAR_FULL)
                continue
            if o["layer"] < player["layer"] and o["type"] == "target" and _covers(player, o):
                hidden.append(o)
                continue
            out.append(o)
        still = []
        for o in hidden:
            (still if _covers(player, o) else out).append(o)
        hidden = still
        out.append(player)
    else:
        out = [o for o in state if o is not bar]
    if bar is not None:
        out.append(bar)
    _mem["last"], _mem["hidden"] = _canon(out), hidden
    return out
