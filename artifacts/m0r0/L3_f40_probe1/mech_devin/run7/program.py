# Mechanics: mirror blocks (block mode) A1/A2 move both blocks y-/+4, A3 apart, A4 together; each block moves alone
#   unless its target 4x4 cell is out of the undrawn playable bound [2,58]^2, an invisible maze cell, a marker cell or the other block.
# Click an inactive marker: blocks become armed players and that marker is active; armed A1-A4 move the active marker by 4
#   (same blocking); clicking another marker switches, clicking a player disarms; A5/A7/other clicks are no-ops.
# Hidden: action counter a (top/bottom bars w=3(a+1)//7), continuity-gated; fallback a=0 if no bar else ceil(7w/3). Unconfirmed: maze cells beyond the 4 inferred.
import json

LO, HI = 2, 58
MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"last": None, "a": 0, "left": None}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _bar_w(a):
    return 3 * (a + 1) // 7


def parse(state):
    m = {"blocks": None, "players": [], "markers": [], "scenery": [], "bar": 0}
    blocks = {}
    for o in state:
        if o["name"] in ("block_left", "block_right"):
            blocks[o["name"]] = (o["x"], o["y"])
        elif o["type"] == "player":
            m["players"].append((o["x"], o["y"]))
        elif o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif "color_0" in o["tags"]:
            m["bar"] = o["w"]
        else:
            m["scenery"].append(o)
    if blocks:
        m["blocks"] = [list(blocks["block_left"]), list(blocks["block_right"])]
    return m


def in_bounds(c):
    return LO <= c[0] <= HI and LO <= c[1] <= HI


def marker_cells(markers, skip=None):
    return {(mk[0] - 1, mk[1] - 1) for i, mk in enumerate(markers) if i != skip}


def free(cell, occupied):
    return in_bounds(cell) and cell not in MAZE and cell not in occupied


def step_blocks(m, act):
    dx, dy = MOVES[act]
    deltas = [(dx, dy), (-dx, dy)]  # left gets dx, right is mirrored
    mc = marker_cells(m["markers"])
    old = [tuple(b) for b in m["blocks"]]
    for i, (ddx, ddy) in enumerate(deltas):
        tgt = (old[i][0] + ddx, old[i][1] + ddy)
        if free(tgt, mc | {old[1 - i]}):
            m["blocks"][i] = list(tgt)


def step_marker(m, act):
    dx, dy = MOVES[act]
    for i, mk in enumerate(m["markers"]):
        if mk[2]:
            tgt = (mk[0] - 1 + dx, mk[1] - 1 + dy)
            occ = marker_cells(m["markers"], skip=i) | set(m["players"])
            if free(tgt, occ):
                mk[0] += dx
                mk[1] += dy


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def click(m, x, y):
    for i, mk in enumerate(m["markers"]):
        if hit(x, y, mk[0], mk[1], 2):
            if mk[2]:
                return
            for k in m["markers"]:
                k[2] = False
            mk[2] = True
            if m["blocks"] is not None:
                _mem["left"] = tuple(m["blocks"][0])
                m["players"] = [tuple(b) for b in m["blocks"]]
                m["blocks"] = None
            return
    if m["blocks"] is None:
        for p in m["players"]:
            if hit(x, y, p[0], p[1], 4):
                ps = sorted(m["players"])
                left = _mem["left"] if _mem["left"] in ps else ps[0]
                right = [p2 for p2 in ps if p2 != left][0]
                m["blocks"] = [list(left), list(right)]
                m["players"] = []
                for k in m["markers"]:
                    k[2] = False
                return


def obj(name, typ, tags, x, y, w, h, layer, pixels=None):
    o = {"name": name, "type": typ, "tags": tags, "x": x, "y": y, "w": w, "h": h,
         "layer": layer, "visible": True}
    if pixels is not None:
        o["pixels"] = pixels
    return o


def render(m, a):
    items = []  # (y, x, builder(prefix_index))
    for s in m["scenery"]:
        col = s["tags"][0].split("_")[1]
        items.append((s["y"], s["x"], s, "%s_%s" % (s["type"], col)))
    w = _bar_w(a)
    if w > 0:
        items.append((0, 64 - w, obj("", "wall", ["color_0", "wall"], 64 - w, 0, w, 1, 0), "wall_0"))
        items.append((63, 0, obj("", "wall", ["color_0", "wall"], 0, 63, w, 1, 0), "wall_0"))
    for mk in m["markers"]:
        tags = ["color_11", "active", "marker"] if mk[2] else ["color_9", "inactive", "marker"]
        items.append((mk[1], mk[0], obj("", "marker", tags, mk[0], mk[1], 2, 2, 1), "marker_" + tags[0][6:]))
    for p in m["players"]:
        items.append((p[1], p[0], obj("", "player", ["color_1", "armed", "player"], p[0], p[1], 4, 4, 1), "player_1"))
    out = []
    start = 0
    if m["blocks"] is not None:
        start = 2
        px = [[10] * 4 for _ in range(4)]
        for nm, b in zip(("block_left", "block_right"), m["blocks"]):
            out.append(obj(nm, "player", ["cyan", "block", "player"], b[0], b[1], 4, 4, 1, [r[:] for r in px]))
    items.sort(key=lambda t: (t[0], t[1]))
    for i, (_, _, o, prefix) in enumerate(items):
        o = dict(o)
        o["name"] = "%s_%d" % (prefix, start + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["last"] is not None and _canon(state) == _mem["last"]:
        a = _mem["a"]
    else:
        a = 0 if m["bar"] == 0 else (7 * m["bar"] + 2) // 3
        _mem["left"] = None
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
    elif action in MOVES:
        if m["blocks"] is not None:
            step_blocks(m, action)
        else:
            step_marker(m, action)
    a += 1
    out = render(m, a)
    _mem["last"] = _canon(out)
    _mem["a"] = a
    return out
