# Mechanics: outline shapes (plus/X/diamond, colour c, radius r) on bg 5 over static boxes (4-ring + colour centre) and swatches (2-border).
# A1-4 move the active shape by 3 (up/down/left/right), blocked only if its centre would leave rows/cols 0..62/63;
# a moved shape touching a swatch bbox takes the swatch colour; A5 passes 'active' to the next shape by size desc (cyclic).
# Active shape draws a 0 at its centre; shapes drawn largest first (smaller on top); occluded bg restored from memory/objects.
# HUD row 63: one '1' added from the right per 4 actions (hidden action counter; stateless fallback assumes 2 into the cycle).
HUD_ROW, BG, STEP = 63, 5, 3
KINDS = ("plus", "X", "diamond")
_mem = {}


def _col(o):
    c = _tag(o, "color_")
    return int(c) if c and c.isdigit() else None


def _tag(o, prefix):
    for t in o.get("tags", []):
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def template(kind, r, cx, cy):
    cells = set()
    for d in range(-r, r + 1):
        if kind == "plus":
            cells |= {(cx + d, cy), (cx, cy + d)}
        elif kind == "X":
            cells |= {(cx + d, cy + d), (cx + d, cy - d)}
        else:
            e = r - abs(d)
            cells |= {(cx + d, cy + e), (cx + d, cy - e)}
    return cells


def in_frame(x, y):
    return 0 <= x < 64 and 0 <= y < HUD_ROW


def shape_cells(s):
    cells = {c for c in template(s["kind"], s["r"], s["cx"], s["cy"]) if in_frame(*c)}
    if s["active"]:
        cells.add((s["cx"], s["cy"]))
    return cells


def match_score(frame, kind, r, cx, cy, col, occluders):
    cells = [c for c in template(kind, r, cx, cy) if in_frame(*c)]
    if not cells:
        return -1
    hit = sum(1 for x, y in cells if frame[y][x] == col)
    hid = sum(1 for x, y in cells if frame[y][x] != col and ((x, y) in occluders or (x, y) == (cx, cy)))
    miss = len(cells) - hit - hid
    return hit + 0.5 * hid - 4 * miss if miss <= 3 else -1


def parse_shapes(state, frame):
    shapes, occluders = [], set()
    players = [o for o in state if o.get("type") == "player"]
    players.sort(key=lambda o: not any(k in o.get("tags", []) for k in KINDS))
    for o in players:
        kinds = [k for k in KINDS if k in o.get("tags", [])]
        col = _col(o) or 0
        if kinds:
            k, r, cx, cy = kinds[0], (o["w"] - 1) // 2, o["cx"], o["cy"]
            r = max(r, (o["h"] - 1) // 2)
        else:  # unclassified sprite: search nearby templates of the missing kinds
            best = (-1,)
            for k in [k for k in KINDS if all(k not in p.get("tags", []) for p in players)] or KINDS:
                for r in range(2, 33):
                    for cx in range(o["cx"] - 6, o["cx"] + 7):
                        for cy in range(o["cy"] - 6, o["cy"] + 7):
                            sc = match_score(frame, k, r, cx, cy, col, occluders)
                            if sc > best[0]:
                                best = (sc, k, r, cx, cy)
            if best[0] < 0:
                continue
            _, k, r, cx, cy = best
        act = in_frame(cx, cy) and frame[cy][cx] == 0
        occluders |= template(k, r, cx, cy)
        shapes.append({"kind": k, "r": r, "cx": cx, "cy": cy, "col": col, "active": act, "tag_active": "active" in o.get("tags", [])})
    if not any(s["active"] for s in shapes):
        for s in shapes:
            others = set().union(*[template(t["kind"], t["r"], t["cx"], t["cy"]) for t in shapes if t is not s])
            s["active"] = s["tag_active"] or (s["cx"], s["cy"]) in others
            if s["active"]:
                break
    for s in shapes:
        s.pop("tag_active")
    return shapes


def statics(state, frame):
    boxes, swatches = [], []
    for o in state:
        col = _col(o)
        rect = (o["x"], o["y"], o["w"], o["h"])
        if o.get("type") == "target":
            boxes.append((rect, col))
        elif o.get("type") == "swatch":
            swatches.append((rect, col))
    return boxes, swatches


def under(x, y, boxes, swatches):
    for (bx, by, bw, bh), col in boxes:
        if bx <= x < bx + bw and by <= y < by + bh:
            if x == bx + bw // 2 and y == by + bh // 2:
                return BG if col is None else col
            return 4
    for (sx, sy, sw, sh), col in swatches:
        if sx <= x < sx + sw and sy <= y < sy + sh:
            border = x in (sx, sx + sw - 1) or y in (sy, sy + sh - 1)
            return 2 if border else col
    return BG


def draw_order(shapes):
    return sorted(range(len(shapes)), key=lambda i: (-shapes[i]["r"], KINDS.index(shapes[i]["kind"])))


def touches(s, rect):
    sx, sy, sw, sh = rect
    return any(sx <= x < sx + sw and sy <= y < sy + sh for x, y in shape_cells(s))


def step_shapes(shapes, action, swatches):
    order = draw_order(shapes)
    act = [i for i in order if shapes[i]["active"]]
    if action == 5 and act:
        cur = order.index(act[0])
        shapes[act[0]]["active"] = False
        shapes[order[(cur + 1) % len(order)]]["active"] = True
    elif action in (1, 2, 3, 4) and act:
        s = shapes[act[0]]
        dx, dy = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}[action]
        if in_frame(s["cx"] + dx, s["cy"] + dy):
            s["cx"] += dx
            s["cy"] += dy
            for rect, col in swatches:
                if col is not None and touches(s, rect):
                    s["col"] = col
    return shapes


def render(bg, shapes, ones):
    out = [row[:] for row in bg]
    for i in draw_order(shapes):
        s = shapes[i]
        for x, y in shape_cells(s):
            out[y][x] = s["col"]
        if s["active"] and in_frame(s["cx"], s["cy"]):
            out[s["cy"]][s["cx"]] = 0
    for k in range(min(ones, 64)):
        out[HUD_ROW][63 - k] = 1
    return out


def transition_function(state, action, frame):
    global _mem
    aid = action.get("action_id") if isinstance(action, dict) else action
    ones = sum(1 for v in frame[HUD_ROW] if v == 1)
    if _mem.get("frame") == frame:
        bg, shapes, count, swatches = _mem["bg"], _mem["shapes"], _mem["count"], _mem["swatches"]
        shapes = [dict(s) for s in shapes]
    else:
        boxes, swatches = statics(state, frame)
        shapes = parse_shapes(state, frame)
        covered = set()
        for s in shapes:
            covered |= shape_cells(s)
        bg = [row[:] for row in frame]
        for x, y in covered:
            bg[y][x] = under(x, y, boxes, swatches)
        for x in range(64):
            if bg[HUD_ROW][x] == 1:
                bg[HUD_ROW][x] = frame[HUD_ROW][0]
        count = 4 * ones + 2
    shapes = step_shapes(shapes, aid, swatches)
    count += 1
    out = render(bg, shapes, count // 4)
    _mem = {"frame": out, "bg": bg, "shapes": [dict(s) for s in shapes], "count": count, "swatches": swatches}
    return out
