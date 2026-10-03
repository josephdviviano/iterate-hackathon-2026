# Mechanics: two cyan 4x4 blocks (4-step grid) move together: A1/A2 = y-4/y+4, A3 = apart, A4 = together (mirrored x);
# each block is blocked by bounds [2,58], invisible maze cells, marker cells (marker xy-1) and the other block.
# Click an inactive marker -> armed mode (blocks -> color_1 players, marker -> color_11 active); A1-A4 then move the
# active marker (same blocking); click active marker/block = no-op; click a player -> back to blocks. A5 = no-op.
# Timer bars (color_0, top-right and bottom-left) have w = 3(a+1)//7, a = actions since level start (hidden, continuity
# gated; fallback a = ceil(7w/3)). Names re-rank by (y,x). Unconfirmed: maze cells only known where moves failed.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
_mem = {"last": None, "a": 0, "left": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _color(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def _hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def parse(state):
    m = {"blocks": None, "players": None, "markers": [], "active": None, "bg": [], "bar": 0}
    blocks, players = {}, []
    for o in state:
        t = o["type"]
        if t == "marker":
            m["markers"].append((o["x"], o["y"]))
            if "active" in o["tags"]:
                m["active"] = (o["x"], o["y"])
        elif t == "player":
            if "block" in o["tags"]:
                blocks[o["name"]] = (o["x"], o["y"])
            else:
                players.append((o["x"], o["y"]))
        elif t == "wall" and _color(o) == 0:
            m["bar"] = o["w"]
        else:
            m["bg"].append(o)
    if blocks:
        m["blocks"] = [blocks["block_left"], blocks["block_right"]]
    else:
        m["players"] = players
    return m


def _free(cell, occupied):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in occupied


def step(m, action, left_pos):
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        if m["blocks"] is not None:
            for mk in m["markers"]:
                if mk[0] <= cx < mk[0] + 2 and mk[1] <= cy < mk[1] + 2:
                    m["players"] = list(m["blocks"])
                    left_pos = m["blocks"][0]
                    m["blocks"] = None
                    m["active"] = mk
                    break
        else:
            for mk in m["markers"]:
                if mk[0] <= cx < mk[0] + 2 and mk[1] <= cy < mk[1] + 2:
                    m["active"] = mk
                    return m, left_pos
            for p in m["players"]:
                if p[0] <= cx < p[0] + 4 and p[1] <= cy < p[1] + 4:
                    ps = m["players"]
                    if left_pos in ps:
                        left = left_pos
                    else:
                        left = min(ps)
                    right = [q for q in ps if q != left]
                    right = right[0] if right else left
                    m["blocks"] = [left, right]
                    m["players"] = None
                    m["active"] = None
                    break
        return m, left_pos
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
    if d is None:
        return m, left_pos
    mcells = {(x - 1, y - 1) for x, y in m["markers"]}
    if m["blocks"] is not None:
        dx, dy = d
        moves = [(dx, dy), (-dx, dy)] if action in (3, 4) else [(dx, dy), (dx, dy)]
        old = list(m["blocks"])
        new = []
        for i, (bx, by) in enumerate(old):
            tgt = (bx + moves[i][0], by + moves[i][1])
            other = {old[1 - i]}
            new.append(tgt if _free(tgt, mcells | other) else (bx, by))
        if new[0] == new[1]:
            new = old
        m["blocks"] = new
    elif m["active"] is not None:
        ax, ay = m["active"]
        tgt = (ax - 1 + d[0], ay - 1 + d[1])
        occ = (mcells - {(ax - 1, ay - 1)}) | set(m["players"])
        if _free(tgt, occ):
            na = (ax + d[0], ay + d[1])
            m["markers"] = [na if mk == m["active"] else mk for mk in m["markers"]]
            m["active"] = na
    return m, left_pos


def render(m, a):
    objs = []
    for o in m["bg"]:
        objs.append(dict(o))
    w = 3 * (a + 1) // 7
    if w > 0:
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                     "w": w, "x": 64 - w, "y": 0})
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True,
                     "w": w, "x": 0, "y": 63})
    for mk in m["markers"]:
        act = mk == m["active"]
        c = 11 if act else 9
        objs.append({"h": 2, "layer": 1, "tags": ["color_%d" % c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": mk[0], "y": mk[1]})
    if m["players"] is not None:
        for p in m["players"]:
            objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                         "visible": True, "w": 4, "x": p[0], "y": p[1]})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    base = 0
    out = []
    if m["blocks"] is not None:
        base = 2
        for nm, (bx, by) in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"h": 4, "layer": 1, "name": nm, "pixels": [r[:] for r in BLOCK_PIX],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": bx, "y": by})
    for i, o in enumerate(objs):
        o["name"] = "%s_%d_%d" % (o["type"], _color(o), base + i)
        out.append(o)
    return out


def transition_function(state, action):
    cont = _mem["last"] is not None and _canon(state) == _mem["last"]
    m = parse(state)
    if cont:
        a, left = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] else 0
        left = None
    m, left = step(m, action, left)
    a += 1
    out = render(m, a)
    _mem.update(last=_canon(out), a=a, left=left)
    return out
