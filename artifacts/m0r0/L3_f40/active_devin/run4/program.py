# Mechanics: block mode = two mirrored cyan 4x4 blocks; A1/A2 move both y-4/+4, A3 apart, A4 together; each block
# moves alone if its target 4x4 cell is in its board half and not a maze/marker/other-block cell. Click inactive marker
# -> armed mode (blocks -> color_1 players, marker active 11); armed A1-A4 move active marker by 4 (blocked by maze,
# markers, players, board); click other marker switches, click player -> block mode; other clicks/A5 no-ops. Bars
# w=floor(3(a+1)/7), a=actions since level start (hidden, continuity-gated; fallback = smallest a). Unconfirmed: unseen maze cells.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
_mem = {"last": None, "a": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _parse(state):
    m = {"blocks": None, "players": None, "markers": [], "bar": 0}
    blocks, players = {}, []
    for o in state:
        t = o.get("tags", [])
        if o["type"] == "marker":
            m["markers"].append([o["x"], o["y"], "active" in t])
        elif "block" in t:
            blocks[o["name"]] = (o["x"], o["y"])
        elif "armed" in t:
            players.append((o["x"], o["y"]))
        elif "color_0" in t and o["y"] == 0:
            m["bar"] = o["w"]
    if blocks:
        m["blocks"] = [list(blocks["block_left"]), list(blocks["block_right"])]
    else:
        m["players"] = sorted(players)
    return m


def _inside(px, py, x, y, w, h):
    return x <= px < x + w and y <= py < y + h


def _marker_cells(m, skip=None):
    return {(mk[0] - 1, mk[1] - 1) for i, mk in enumerate(m["markers"]) if i != skip}


def _block_free(m, idx, cell, other):
    x, y = cell
    xlo, xhi = (LO, 26) if idx == 0 else (34, HI)
    if not (xlo <= x <= xhi and LO <= y <= HI):
        return False
    return cell not in MAZE and cell not in _marker_cells(m) and cell != tuple(other)


def _move_blocks(m, act):
    d = {1: [(0, -4), (0, -4)], 2: [(0, 4), (0, 4)], 3: [(-4, 0), (4, 0)], 4: [(4, 0), (-4, 0)]}[act]
    old = [tuple(b) for b in m["blocks"]]
    new = []
    for i in range(2):
        c = (old[i][0] + d[i][0], old[i][1] + d[i][1])
        new.append(list(c) if _block_free(m, i, c, old[1 - i]) else list(old[i]))
    m["blocks"] = new


def _move_marker(m, act):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[act]
    for i, mk in enumerate(m["markers"]):
        if mk[2]:
            c = (mk[0] - 1 + dx, mk[1] - 1 + dy)
            if (LO <= c[0] <= HI and LO <= c[1] <= HI and c not in MAZE
                    and c not in _marker_cells(m, skip=i) and c not in set(m["players"])):
                mk[0], mk[1] = c[0] + 1, c[1] + 1


def _click(m, x, y):
    hit = None
    for i, mk in enumerate(m["markers"]):
        if _inside(x, y, mk[0], mk[1], 2, 2):
            hit = i
    if hit is not None:
        if m["markers"][hit][2]:
            return
        for i, mk in enumerate(m["markers"]):
            mk[2] = i == hit
        if m["blocks"] is not None:
            m["players"] = sorted(tuple(b) for b in m["blocks"])
            m["blocks"] = None
        return
    if m["players"] is not None:
        for p in m["players"]:
            if _inside(x, y, p[0], p[1], 4, 4):
                m["blocks"] = [list(q) for q in sorted(m["players"])]
                m["players"] = None
                for mk in m["markers"]:
                    mk[2] = False
                return


def _step(m, action):
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            _click(m, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if m["blocks"] is not None:
            _move_blocks(m, action)
        else:
            _move_marker(m, action)


def _render(m, a):
    objs = []
    bw = (3 * (a + 1)) // 7
    base = {"layer": 0, "visible": True}
    if bw > 0:
        objs.append(dict(base, type="wall", tags=["color_0", "wall"], c="0", x=64 - bw, y=0, w=bw, h=1))
        objs.append(dict(base, type="wall", tags=["color_0", "wall"], c="0", x=0, y=63, w=bw, h=1))
    objs.append(dict(base, type="wall", tags=["color_15", "wall"], c="15", x=0, y=1, w=32, h=62))
    objs.append(dict(base, type="hazard", tags=["color_8", "hazard"], c="8", x=32, y=1, w=32, h=62))
    for mx, my, act in m["markers"]:
        col = "11" if act else "9"
        objs.append({"type": "marker", "tags": ["color_" + col, "active" if act else "inactive", "marker"],
                     "c": col, "x": mx, "y": my, "w": 2, "h": 2, "layer": 1, "visible": True})
    if m["players"] is not None:
        for px, py in m["players"]:
            objs.append({"type": "player", "tags": ["color_1", "armed", "player"], "c": "1",
                         "x": px, "y": py, "w": 4, "h": 4, "layer": 1, "visible": True})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    out = []
    start = 2 if m["blocks"] is not None else 0
    for i, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"], o.pop("c"), start + i)
        out.append(o)
    if m["blocks"] is not None:
        for nm, (bx, by) in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"name": nm, "type": "player", "tags": ["cyan", "block", "player"], "x": bx, "y": by,
                        "w": 4, "h": 4, "layer": 1, "visible": True, "pixels": [r[:] for r in BLOCK_PIX]})
    return out


def transition_function(state, action):
    m = _parse(state)
    if _mem["last"] is not None and _canon(state) == _mem["last"]:
        a = _mem["a"]
    else:
        a = (7 * m["bar"] - 1) // 3 if m["bar"] > 0 else 0
    _step(m, action)
    a += 1
    out = _render(m, a)
    _mem["last"], _mem["a"] = _canon(out), a
    return out
