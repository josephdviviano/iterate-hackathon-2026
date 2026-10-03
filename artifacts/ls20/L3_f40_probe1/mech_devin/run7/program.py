# Mechanics: player 5x5 steps 5 cells (A1-4 = up/down/left/right); invisible maze walls {(39,5),(34,10)} block
# moves but still burn budget; portal: entering (9,5) lands at (34,5). Undrawn playable-area bound: rows y>=50
# (HUD band) and the 64x64 board edge -> a move leaving it is refused entirely (no move, no budget spent).
# step_bar burns 2 per action (x = 55 - w, tag = str(w)); a fully covered refuel ring is consumed, bar -> 42.
# Covered layer-0 targets are occluded and restored when uncovered (continuity-gated memory). Unconfirmed: other bounds.
import copy, json

STEP = 5
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BOUND = (0, 0, 64, 50)  # x0, y0, x1, y1 of the playable area (exclusive right/bottom)
BURN, FULL, BAR_RIGHT = 2, 42, 55
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}

_mem = {"last": None, "hidden": []}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"] and
            o["x"] + o["w"] <= p["x"] + p["w"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _inside_bound(x, y, w, h):
    x0, y0, x1, y1 = BOUND
    return x >= x0 and y >= y0 and x + w <= x1 and y + h <= y1


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def _step(state, action, hidden):
    s = copy.deepcopy(state)
    hidden = copy.deepcopy(hidden)
    player = next((o for o in s if o["type"] == "player"), None)
    bar = next((o for o in s if o["name"] == "step_bar"), None)
    aid = action["action_id"] if isinstance(action, dict) else action
    if player is not None and aid in MOVES:
        dx, dy = MOVES[aid]
        nx, ny = player["x"] + dx, player["y"] + dy
        if not _inside_bound(nx, ny, player["w"], player["h"]):
            return s, hidden  # outside the undrawn playable bound: refused, free
        if (nx, ny) not in WALLS:
            nx, ny = PORTALS.get((nx, ny), (nx, ny))
            player["x"], player["y"] = nx, ny
    if bar is not None:
        _set_bar(bar, bar["w"] - BURN)
    if player is None:
        return s, hidden
    # restore uncovered occluded objects
    keep = []
    for o in hidden:
        if _covers(player, o):
            keep.append(o)
        else:
            s.append(o)
    hidden = keep
    out = []
    for o in s:
        if o is player or o["layer"] != 0 or not _covers(player, o):
            out.append(o)
        elif o["type"] == "refuel":
            if bar is not None:
                _set_bar(bar, FULL)
        elif o["type"] == "target":
            hidden.append(o)
        else:
            out.append(o)
    return out, hidden


def transition_function(state, action):
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    out, hidden = _step(state, action, hidden)
    _mem["last"], _mem["hidden"] = _canon(out), hidden
    return out
