# Mechanics: two mirror axes (h-axis = 3-row wall band, v-axis = 3-col wall band) and pieces made of 3x3 blocks;
# ACTION5 cycles selection h-axis -> v-axis -> piece sprites -> h-axis; selected holes render 0. ACTION1/2 move h-axis
# or piece by x-/+3, ACTION3/4 move v-axis or piece by y-/+3 (wrong-direction axis move = no-op); blocked by bounds,
# other pieces, axis bands. v-axis reflections drawn gray (4); h/both reflections invisible occluders (unconfirmed).
# Hypotheses: sprite order for ACTION5 (descending x,y), piece-vs-axis blocking, invisible reflections: unobserved.
N = 63
NB = 21
WALL, TARGET, REFL, PIECE = 1, 2, 3, 4
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memo = {"out": None, "model": None}


def _canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(o):
    out = {}
    p = o.get("pixels")
    if not p:
        return out
    for i in range(o["w"]):
        for j in range(o["h"]):
            if p[i][j] >= 0:
                out[(o["x"] + i, o["y"] + j)] = p[i][j]
    return out


def _groups(blocks, diag):
    blocks, seen, res = set(blocks), set(), []
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for b in sorted(blocks):
        if b in seen:
            continue
        comp, stack = set(), [b]
        seen.add(b)
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in nb:
                n = (r + dr, c + dc)
                if n in blocks and n not in seen:
                    seen.add(n)
                    stack.append(n)
        res.append(comp)
    return res


def parse(state):
    wall, piece, target, extra = {}, {}, {}, []
    for o in state:
        t = o["type"]
        if t == "wall":
            wall.update(_cells(o))
        elif t == "player":
            piece.update(_cells(o))
        elif t == "target":
            target.update(_cells(o))
        elif t not in ("reflection",):
            extra.append(o)
    rows, cols = {}, {}
    for (r, c) in {(r // 3, c // 3) for (r, c) in wall}:
        rows[r] = rows.get(r, 0) + 1
        cols[c] = cols.get(c, 0) + 1
    A = max(rows, key=lambda k: (rows[k], -k)) if rows else 0
    B = max(cols, key=lambda k: (cols[k], -k)) if cols else 0
    pblocks = {(r // 3, c // 3) for (r, c) in piece}
    sprites = _groups(pblocks, True)
    sel = None
    if any(wall.get((3 * A + 1, 3 * k + 1)) == 0 for k in range(NB) if k != B):
        sel = "h"
    elif any(wall.get((3 * k + 1, 3 * B + 1)) == 0 for k in range(NB) if k != A):
        sel = "v"
    for i, s in enumerate(sprites):
        if any(piece.get((3 * r + 1, 3 * c + 1)) == 0 for (r, c) in s):
            sel = i
    tblocks = {(r // 3, c // 3) for (r, c) in target}
    return {"A": A, "B": B, "sprites": [sorted(s) for s in sprites], "sel": sel,
            "targets": sorted(tblocks), "extra": extra}


def order(model):
    return sorted(range(len(model["sprites"])), key=lambda i: min(model["sprites"][i]), reverse=True)


def step(model, action):
    m = dict(model)
    m["sprites"] = [list(s) for s in model["sprites"]]
    aid = action["action_id"] if isinstance(action, dict) else action
    sel = m["sel"]
    if aid == 5:
        cyc = ["h", "v"] + order(m)
        m["sel"] = cyc[(cyc.index(sel) + 1) % len(cyc)] if sel in cyc else "h"
        return m
    if aid not in DIRS:
        return m
    dr, dc = DIRS[aid]
    allp = {b for s in m["sprites"] for b in s}
    if sel == "h" and dr:
        a = m["A"] + dr
        if 0 <= a < NB and not any(r == a for (r, c) in allp):
            m["A"] = a
    elif sel == "v" and dc:
        b = m["B"] + dc
        if 0 <= b < NB and not any(c == b for (r, c) in allp):
            m["B"] = b
    elif isinstance(sel, int):
        mine = set(m["sprites"][sel])
        others = allp - mine
        new = [(r + dr, c + dc) for (r, c) in m["sprites"][sel]]
        if all(0 <= r < NB and 0 <= c < NB and (r, c) not in others
               and r != m["A"] and c != m["B"] for (r, c) in new):
            m["sprites"][sel] = sorted(new)
    return m


def _block(sprite, blocks, solid, hole):
    for (br, bc) in blocks:
        for i in range(3):
            for j in range(3):
                cell = (3 * br + i, 3 * bc + j)
                if i == 1 and j == 1:
                    sprite["cells"][cell] = ("h", hole)
                else:
                    sprite["cells"][cell] = ("s", solid)


def build(m):
    A, B, sel = m["A"], m["B"], m["sel"]
    sp = []
    h = {"type": "wall", "layer": WALL, "cells": {}}
    _block(h, [(A, k) for k in range(NB)], 10, 0 if sel == "h" else -1)
    v = {"type": "wall", "layer": WALL, "cells": {}}
    _block(v, [(k, B) for k in range(NB)], 10, 0 if sel == "v" else -1)
    sp += [h, v]
    t = {"type": "target", "layer": TARGET, "cells": {}}
    _block(t, m["targets"], 11, 11)
    for c in list(t["cells"]):
        t["cells"][c] = ("s", 11)
    sp.append(t)
    allp = {b for s in m["sprites"] for b in s}
    vis, inv = set(), set()
    for (r, c) in allp:
        for (rr, cc), visible in (((r, 2 * B - c), True), ((2 * A - r, c), False), ((2 * A - r, 2 * B - c), False)):
            if 0 <= rr < NB and 0 <= cc < NB and (rr, cc) not in allp:
                (vis if visible else inv).add((rr, cc))
    inv -= vis
    rv = {"type": "reflection", "layer": REFL, "cells": {}}
    _block(rv, vis, 4, 4)
    ri = {"type": "invisible", "layer": REFL, "cells": {}}
    _block(ri, inv, None, -1)
    sp += [rv, ri]
    for i, s in enumerate(m["sprites"]):
        p = {"type": "player", "layer": PIECE, "cells": {}}
        _block(p, s, 5, 0 if sel == i else -1)
        sp.append(p)
    return sp, inv


def composite(sprites):
    owner = {}
    stack = {}
    for k, s in enumerate(sprites):
        for cell, e in s["cells"].items():
            if 0 <= cell[0] < N and 0 <= cell[1] < N:
                stack.setdefault(cell, []).append((s["layer"], k, e))
    for cell, ents in stack.items():
        ents.sort(key=lambda e: -e[0])
        solid = [e for e in ents if e[2][0] == "s"]
        if solid:
            lay, k, (_, val) = solid[0]
            if val is not None:
                owner[cell] = (sprites[k]["type"], val)
            continue
        top = ents[0][0]
        best = max((e for e in ents if e[0] == top), key=lambda e: e[2][1])
        if best[2][1] >= 0:
            owner[cell] = (sprites[best[1]]["type"], best[2][1])
    return owner


def _comps(cells):
    cells, seen, res = set(cells), set(), []
    for c in sorted(cells):
        if c in seen:
            continue
        comp, stack = set(), [c]
        seen.add(c)
        while stack:
            r, q = stack.pop()
            comp.add((r, q))
            for n in ((r + 1, q), (r - 1, q), (r, q + 1), (r, q - 1)):
                if n in cells and n not in seen:
                    seen.add(n)
                    stack.append(n)
        res.append(comp)
    return res


def _bbox(comp):
    xs = [r for r, c in comp]
    ys = [c for r, c in comp]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_close(comps):
    comps = [set(c) for c in comps]
    changed = True
    while changed:
        changed = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = _bbox(comps[i]), _bbox(comps[j])
                gx = max(b[0] - a[2], a[0] - b[2]) - 1
                gy = max(b[1] - a[3], a[1] - b[3]) - 1
                if gx <= 3 and gy <= 3:
                    comps[i] |= comps.pop(j)
                    changed = True
                    break
            if changed:
                break
    return comps


def _obj(comp, owner, layer, tags, typ):
    x0, y0, x1, y1 = _bbox(comp)
    pix = [[owner[(x, y)][1] if (x, y) in comp else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {"type": typ, "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "pixels": pix, "layer": layer, "tags": tags}


def _zero_keys(owner, typ):
    return [_bbox(c)[:2] for c in _comps([k for k, v in owner.items() if v[0] == typ and v[1] == 0])]


def render(m):
    sprites, inv = build(m)
    owner = composite(sprites)
    by = lambda t: [c for c, v in owner.items() if v[0] == t]
    out = [dict(o) for o in m["extra"]]
    walls = _comps(by("wall"))
    keys = sorted([(_bbox(c)[:2], 0, c) for c in walls] + [(k, 1, None) for k in _zero_keys(owner, "player")],
                  key=lambda e: e[0])
    for rank, (_, kind, c) in enumerate(keys):
        if kind == 0:
            o = _obj(c, owner, WALL, None, "wall")
            ori = "horizontal" if o["h"] > o["w"] else "vertical"
            o["tags"] = ["axis", ori]
            o["name"] = "wall_%s_%d" % (ori[0], rank)
            out.append(o)
    pieces = _comps(by("player"))
    keys = sorted([(_bbox(c)[:2], 0, c) for c in pieces] + [(k, 1, None) for k in _zero_keys(owner, "wall")],
                  key=lambda e: e[0])
    for rank, (_, kind, c) in enumerate(keys):
        if kind == 0:
            o = _obj(c, owner, PIECE, ["movable", "black"], "player")
            o["name"] = "piece_%d" % rank
            out.append(o)
    invc = [{(3 * r + i, 3 * q + j) for (r, q) in g for i in range(3) for j in range(3)} for g in _groups(inv, False)]
    keys = sorted([(_bbox(c)[:2], 0, c) for c in _comps(by("reflection"))] + [(_bbox(c)[:2], 1, None) for c in invc],
                  key=lambda e: e[0])
    for rank, (_, kind, c) in enumerate(keys):
        if kind == 0:
            o = _obj(c, owner, REFL, ["mirror", "gray"], "reflection")
            o["name"] = "reflection_%d" % rank
            out.append(o)
    tg = sorted(_merge_close(_comps(by("target"))), key=lambda c: _bbox(c)[:2])
    for rank, c in enumerate(tg):
        o = _obj(c, owner, TARGET, ["goal", "yellow"], "target")
        o["name"] = "target_%d" % rank
        out.append(o)
    return out


def transition_function(state, action):
    model = None
    if _memo["out"] is not None and _memo["out"] == _canon(state):
        model = _memo["model"]
    if model is None:
        model = parse(state)
        if model["sel"] is None:
            model["sel"] = "h"
    m = step(model, action)
    out = render(m)
    _memo["out"], _memo["model"] = _canon(out), m
    return out
