# Mechanics: 3x3-cell mirror puzzle. A horizontal wall band (axis) mirrors every piece into a gray reflection.
# ACTION5 cycles selection axis -> piece_0 -> piece_1 ... -> axis; ACTION1/2/3/4 move the selection up/down/left/right
# by one cell (axis only moves vertically). Selected sprite shows black(0) cell-centre dots; every cell-centre hole of
# pieces/reflections is see-through onto targets (targets occluded elsewhere). Frame is re-rendered and re-extracted:
# 4-connected components per type, targets per sprite, wall_h index offset by #black piece pixels. Hypotheses: blocking rules.
import copy, json

N = 63
TAGS = {"player": ["movable", "black"], "reflection": ["mirror", "gray"],
        "target": ["goal", "yellow"], "wall": ["axis", "horizontal"]}
LAYER = {"wall": 1, "target": 2, "reflection": 3, "player": 4}
_memo = {"out": None, "model": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def obj_pixels(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o["x"] + i, o["y"] + j, v


def parse(state):
    """Recover the cell-level model from an observed frame."""
    m = {"A": None, "sel": None, "pieces": [], "targets": [], "other": []}
    sel_axis = False
    for o in state:
        t = o.get("type")
        if t == "wall":
            m["A"] = o["x"] if m["A"] is None else min(m["A"], o["x"])
            sel_axis |= any(v == 0 for _, _, v in obj_pixels(o))
        elif t == "target":
            m["targets"].append(sorted({(r // 3, c // 3) for r, c, v in obj_pixels(o)}))
        elif t not in ("player", "reflection"):
            m["other"].append(copy.deepcopy(o))
    tcells = {c for tg in m["targets"] for c in tg}
    sel_group, groups = [], []
    for o in sorted((o for o in state if o.get("type") == "player"), key=lambda o: (o["x"], o["y"])):
        px = {(r, c): v for r, c, v in obj_pixels(o)}
        cells = {(r // 3, c // 3) for r, c in px}
        cls = {}
        for R, C in cells:
            centre = px.get((3 * R + 1, 3 * C + 1), -1)
            if centre == 0:
                cls[(R, C)] = 1
            elif (R, C) not in tcells:
                cls[(R, C)] = 0
        todo = [c for c in cells if c not in cls]
        changed = True
        while todo and changed:
            changed = False
            for c in list(todo):
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (c[0] + d[0], c[1] + d[1])
                    if nb in cls:
                        cls[c] = cls[nb]; todo.remove(c); changed = True; break
        for c in todo:
            cls[c] = 0
        sel = [c for c in cells if cls[c] == 1]
        if sel:
            sel_group += sel
        for comp in components(set(c for c in cells if cls[c] == 0)):
            groups.append(sorted(comp))
    if sel_group:
        groups.append(sorted(sel_group))
    groups.sort(key=lambda g: min(g))
    m["pieces"] = groups
    if sel_group and not sel_axis:
        m["sel"] = groups.index(sorted(sel_group))
    return m


def components(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c = stack.pop(); comp.add(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in cells:
                    cells.remove(nb); stack.append(nb)
        out.append(comp)
    return out


def render(m):
    """Frame: dict pixel -> (type, sprite id, value)."""
    f = {}

    def put(r, c, owner, v):
        if 0 <= r < N and 0 <= c < N + 1:
            f[(r, c)] = (owner[0], owner[1], v)

    def stamp(R, C, owner, val, hole_fallback):
        for i in range(3):
            for j in range(3):
                r, c = 3 * R + i, 3 * C + j
                if (i, j) == (1, 1):
                    if (r, c) in f:
                        continue
                    if hole_fallback is not None:
                        put(r, c, owner, hole_fallback)
                else:
                    put(r, c, owner, val)

    A = m["A"]
    if A is not None:
        for i in range(3):
            for c in range(N):
                if i == 1 and c % 3 == 1:
                    if m["sel"] is None:
                        put(A + i, c, ("wall", 0), 0)
                else:
                    put(A + i, c, ("wall", 0), 10)
    for k, tg in enumerate(m["targets"]):
        for R, C in tg:
            for i in range(3):
                for j in range(3):
                    put(3 * R + i, 3 * C + j, ("target", k), 11)
    if A is not None:
        axis2 = 2 * A // 3   # mirrored cell row = axis2 - R
        for p in m["pieces"]:
            for R, C in p:
                stamp(axis2 - R, C, ("reflection", 0), 4, 4)
    for k, p in enumerate(m["pieces"]):
        for R, C in p:
            stamp(R, C, ("player", 0), 5, 0 if m["sel"] == k else None)
    return f


def mk(typ, pix, name):
    xs = [r for r, c in pix]; ys = [c for r, c in pix]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x + 1, max(ys) - y + 1
    grid = [[-1] * h for _ in range(w)]
    for (r, c), v in pix.items():
        grid[r - x][c - y] = v
    return {"name": name, "type": typ, "tags": list(TAGS[typ]), "layer": LAYER[typ],
            "x": x, "y": y, "w": w, "h": h, "pixels": grid}


def extract(f, m):
    out = [copy.deepcopy(o) for o in m["other"]]
    by = {}
    for p, (t, k, v) in f.items():
        by.setdefault(t, {})[p] = v
    black = sum(1 for p, (t, k, v) in f.items() if v == 0 and t != "wall")
    for typ, prefix, off in (("player", "piece", 0), ("reflection", "reflection", 0), ("wall", "wall_h", black)):
        pix = by.get(typ, {})
        objs = []
        for comp in components(set(pix)):
            objs.append({p: pix[p] for p in comp})
        objs = [mk(typ, o, "") for o in objs]
        objs.sort(key=lambda o: (o["x"], o["y"]))
        for i, o in enumerate(objs):
            o["name"] = "%s_%d" % (prefix, off + i)
        out += objs
    tobjs = []
    for k in range(len(m["targets"])):
        pix = {p: v for p, (t, kk, v) in f.items() if t == "target" and kk == k}
        if pix:
            tobjs.append(mk("target", pix, ""))
    tobjs.sort(key=lambda o: (o["x"], o["y"]))
    for i, o in enumerate(tobjs):
        o["name"] = "target_%d" % i
    return out + tobjs


DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def step(m, action):
    m = copy.deepcopy(m)
    aid = action.get("action_id") if isinstance(action, dict) else action
    if aid == 5:
        n = len(m["pieces"])
        if m["sel"] is None:
            m["sel"] = 0 if n else None
        else:
            m["sel"] = m["sel"] + 1 if m["sel"] + 1 < n else None
    elif aid in DIRS:
        dr, dc = DIRS[aid]
        if m["sel"] is None:
            if m["A"] is not None and dr and 0 <= m["A"] + 3 * dr <= N - 3:
                m["A"] += 3 * dr
        else:
            k = m["sel"]
            moved = [(R + dr, C + dc) for R, C in m["pieces"][k]]
            others = {c for i, p in enumerate(m["pieces"]) if i != k for c in p}
            if all(0 <= R < N // 3 and 0 <= C < N // 3 and (R, C) not in others for R, C in moved):
                m["pieces"][k] = moved
    return m


def transition_function(state, action):
    key = canon(state)
    if _memo["out"] is not None and key == _memo["out"]:
        m = _memo["model"]
    else:
        m = parse(state)
    m2 = step(m, action)
    out = extract(render(m2), m2)
    _memo["out"], _memo["model"] = canon(out), m2
    return out
