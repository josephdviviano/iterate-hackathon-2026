# Mechanics: player (5x5, layer 1) moves 5px per A1-A4 (up/down/left/right); hidden walls {(39,5),(34,10)} block,
# portal cell (9,5) warps to (34,5). Every action costs step_bar 2 (w-2, x+2 keeps right edge 55, tag=str(w)).
# Refuel ring fully covered by the player is consumed and refills step_bar to 42. Other layer-0 objects fully
# covered are occluded (not extracted) and reappear when uncovered (continuity-gated memory). Guard: a move into a
# 'chamber' object while hud_glyph is 'pending' (locked exit) is rejected, costs nothing. Unconfirmed: what unlocks it.
import copy
import json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_FULL, BAR_RIGHT, BAR_COST = 42, 55, 2

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"]
            and o["x"] + o["w"] <= p["x"] + p["w"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _box(x, y, p):
    return {"x": x, "y": y, "w": p["w"], "h": p["h"]}


def _locked(state):
    return any("pending" in o.get("tags", []) for o in state if o["name"] == "hud_glyph")


def _blocked_by_chamber(state, dest):
    return _locked(state) and any("chamber" in o.get("tags", []) and _covers(dest, o) for o in state)


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t for t in bar["tags"] if not t.isdigit()] + [str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    hidden = copy.deepcopy(hidden)
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in state if o["type"] == "player"), None)
    bar = next((o for o in state if o["name"] == "step_bar"), None)

    if player is not None and aid in MOVES:
        dx, dy = MOVES[aid]
        nx, ny = player["x"] + dx, player["y"] + dy
        if (nx, ny) in PORTALS:
            nx, ny = PORTALS[(nx, ny)]
        if _blocked_by_chamber(state + hidden, _box(nx, ny, player)):
            _mem["last"], _mem["hidden"] = _canon(state), hidden
            return state
        if (nx, ny) not in WALLS:
            player["x"], player["y"] = nx, ny

    if bar is not None:
        _set_bar(bar, bar["w"] - BAR_COST)

    out = []
    pool = [o for o in state if o is not player] + hidden
    new_hidden = []
    for o in pool:
        if player is not None and o["layer"] < player["layer"] and _covers(player, o):
            if o["type"] == "refuel":
                if bar is not None:
                    _set_bar(bar, BAR_FULL)
                continue
            new_hidden.append(o)
            continue
        out.append(o)
    if player is not None:
        out.append(player)
    _mem["last"], _mem["hidden"] = _canon(out), new_hidden
    return out
