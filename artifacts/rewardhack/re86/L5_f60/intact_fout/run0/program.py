"""Mechanics: actions 1-4 move the active shape 3 cells up/down/left/right; action 5 passes "active" to the next shape.
Shapes are outlines (X, plus, diamond) drawn over a floor of colour 5 with target boxes and swatches; draw order
plus < X < diamond, and the active shape shows a colour-0 centre pixel. A shape whose pixels touch a swatch takes the
swatch colour. HUD row 63 fills one cell with colour 1 from the right every 4 actions.
Shapes, boxes and swatches carry over between continuous calls; "raw" shapes are located on the frame by template.
Unconfirmed: HUD phase on cold start (assumed 2 actions after a tick); screen-edge limits; the order of action 5.
"""

FLOOR, CENTRE, HUD_ON = 5, 0, 1
BOX_EDGE, SWATCH_EDGE = 4, 2
DRAW_ORDER = {"plus": 0, "X": 1, "diamond": 2}
MOVES = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
HUD_PERIOD, COLD_PHASE = 4, 2

_memory = {"frame": None, "phase": 0, "shapes": [], "boxes": [], "swatches": []}


def _colour(obj):
    for t in obj["tags"]:
        if t.startswith("color_"):
            return int(t[6:])
    return FLOOR


def _template(kind, w, h):
    cx, cy = w // 2, h // 2
    if kind == "X":
        return [(i, i) for i in range(w)] + [(w - 1 - i, i) for i in range(w)]
    if kind == "plus":
        return [(i, cy) for i in range(w)] + [(cx, j) for j in range(h)]
    return [(cx + dx, cy + dy) for dx in range(-cx, cx + 1)
            for dy in range(-cy, cy + 1) if abs(dx) + abs(dy) == cx]


def _tagged_kind(obj):
    return next((k for k in DRAW_ORDER if k in obj["tags"]), None)


def _locate(obj, kind, frame):
    """Best-matching placement and size of an unclassified shape near its reported box."""
    col, best = _colour(obj), None
    for size in range(5, 33, 2):
        tpl = _template(kind, size, size)
        for y in range(obj["y"] - size, obj["y"] + obj["h"] + 1):
            for x in range(obj["x"] - size, obj["x"] + obj["w"] + 1):
                score = 0
                for px, py in tpl:
                    if _inside(x + px, y + py):
                        score += 1 if frame[y + py][x + px] in (col, CENTRE) else -1
                if best is None or score > best[0]:
                    best = (score, x, y, size)
    return best


def _parse_shapes(state, frame):
    players = [o for o in state if o["type"] == "player"]
    shapes, unknown = [], []
    for o in players:
        kind = _tagged_kind(o)
        if kind is None:
            unknown.append(o)
            continue
        shapes.append({"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"], "kind": kind,
                       "colour": _colour(o), "active": "active" in o["tags"]})
    for o in unknown:
        free = [k for k in DRAW_ORDER if all(s["kind"] != k for s in shapes)] or list(DRAW_ORDER)
        score, x, y, size, kind = max(_locate(o, k, frame) + (k,) for k in free)
        shapes.append({"x": x, "y": y, "w": size, "h": size, "kind": kind,
                       "colour": _colour(o), "active": "active" in o["tags"]})
    if sum(s["active"] for s in shapes) != 1:
        for s in shapes:
            cx, cy = s["x"] + s["w"] // 2, s["y"] + s["h"] // 2
            s["active"] = _inside(cx, cy) and frame[cy][cx] == CENTRE
    return shapes


def _inside(x, y):
    return 0 <= x < 64 and 0 <= y < 63


def _pixels(shape):
    return [(shape["x"] + px, shape["y"] + py)
            for px, py in _template(shape["kind"], shape["w"], shape["h"])
            if _inside(shape["x"] + px, shape["y"] + py)]


def _touches(shape, swatch):
    return any(swatch["x"] <= px < swatch["x"] + swatch["w"] and swatch["y"] <= py < swatch["y"] + swatch["h"]
               for px, py in _pixels(shape))


def _update_shapes(shapes, swatches, action):
    active = next((i for i, s in enumerate(shapes) if s["active"]), None)
    if active is None:
        return
    if action in MOVES:
        s = shapes[active]
        s["x"] += MOVES[action][0]
        s["y"] += MOVES[action][1]
        for sw in swatches:
            if _touches(s, sw):
                s["colour"] = _colour(sw)
    elif action == 5:
        order = sorted(range(len(shapes)), key=lambda i: DRAW_ORDER[shapes[i]["kind"]])
        nxt = order[(order.index(active) + 1) % len(order)]
        shapes[active]["active"] = False
        shapes[nxt]["active"] = True


def _render_floor(out, boxes, swatches):
    for y in range(63):
        out[y] = [FLOOR] * 64
    for obj, edge in [(b, BOX_EDGE) for b in boxes] + [(s, SWATCH_EDGE) for s in swatches]:
        x0, y0, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                if _inside(x, y):
                    border = x in (x0, x0 + w - 1) or y in (y0, y0 + h - 1)
                    out[y][x] = edge if border else _colour(obj)


def _render_shapes(out, shapes):
    for s in sorted(shapes, key=lambda s: DRAW_ORDER[s["kind"]]):
        for px, py in _pixels(s):
            out[py][px] = s["colour"]
        if s["active"]:
            cx, cy = s["x"] + s["w"] // 2, s["y"] + s["h"] // 2
            if _inside(cx, cy):
                out[cy][cx] = CENTRE


def _tick_hud(out):
    row = out[63]
    filled = sum(1 for c in row if c == HUD_ON)
    if filled < 64:
        row[63 - filled] = HUD_ON


def transition_function(state, action, frame):
    if frame == _memory["frame"]:
        phase, shapes = _memory["phase"], _memory["shapes"]
        boxes, swatches = _memory["boxes"], _memory["swatches"]
    else:
        phase, shapes = COLD_PHASE, _parse_shapes(state, frame)
        boxes = [o for o in state if o["type"] == "target"]
        swatches = [o for o in state if o["type"] == "swatch"]
    act = action["action_id"] if isinstance(action, dict) else action


    _update_shapes(shapes, swatches, act)
    out = [row[:] for row in frame]
    _render_floor(out, boxes, swatches)
    _render_shapes(out, shapes)

    phase = (phase + 1) % HUD_PERIOD
    if phase == 0:
        _tick_hud(out)

    _memory.update(frame=[row[:] for row in out], phase=phase, shapes=[dict(s) for s in shapes],
                   boxes=boxes, swatches=swatches)
    return out
