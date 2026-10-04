# A 21x21 lattice uses 3x3 tiles: axis 10, pieces 5, targets 11, reflections 4.
# A1/A2 move the selected axis or piece vertically; A3/A4 move pieces horizontally.
# Pieces stop at board edges or other pieces; A5 cycles axis then pieces by size.
# Reflections lie at row 2*axis-row; solid centres show through layered tile holes.
# HUD gains one cell per effective action; clicks, undo and exhausted-budget rules are unconfirmed.
N = 21
BG, WALL, TGT, REFL, PIECE, DOT, HUD_ON = 9, 10, 11, 4, 5, 0, 12


def block(frame, bx, by):
    return [frame[3 * by + j][3 * bx + i] for j in range(3) for i in range(3)]


def components(cells):
    cells, out = set(cells), []
    while cells:
        stack = [cells.pop()]
        comp = set(stack)
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(frame):
    walls, targets, pieces = {}, set(), {}
    for by in range(N):
        for bx in range(N):
            px = block(frame, bx, by)
            corner, centre = px[0], px[4]
            if TGT in px:
                targets.add((bx, by))
            if corner == WALL:
                walls[(bx, by)] = centre
            elif corner == PIECE:
                pieces[(bx, by)] = centre
    rows = {}
    for bx, by in walls:
        rows[by] = rows.get(by, 0) + 1
    axis = max(rows, key=rows.get) if rows else 0
    axis_selected = any(c == DOT for c in walls.values())
    groups, selected = [], None
    for comp in components(pieces):
        dots = {b for b in comp if pieces[b] == DOT} if not axis_selected else set()
        if not dots:
            groups.append(comp)
            continue
        grown, stack = set(dots), list(dots)
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in comp and n not in grown and pieces[n] != BG:
                    grown.add(n)
                    stack.append(n)
        selected = len(groups)
        groups.append(grown)
        groups.extend(components(comp - grown))
    return {"axis": axis, "selected": None if axis_selected else selected,
            "pieces": groups, "targets": targets}


def inside_board(cells):
    return all(0 <= x < N and 0 <= y < N for x, y in cells)


def blocked_by_piece(cells, pieces, selected):
    return any(cells & p for i, p in enumerate(pieces) if i != selected)


def update_wall(model, dx, dy):
    if dx or not 0 <= model["axis"] + dy < N:
        return False
    model["axis"] += dy
    return True


def update_player(model, dx, dy):
    selected = model["selected"]
    moved = {(x + dx, y + dy) for x, y in model["pieces"][selected]}
    if not inside_board(moved) or blocked_by_piece(moved, model["pieces"], selected):
        return False
    model["pieces"][selected] = moved
    return True


def update_selection(model):
    order = sorted(range(len(model["pieces"])),
                   key=lambda i: (-len(model["pieces"][i]), min(model["pieces"][i])))
    sequence = [None] + order
    old = model["selected"]
    model["selected"] = sequence[(sequence.index(old) + 1) % len(sequence)]
    return model["selected"] != old


def update_reflection(model):
    axis = model["axis"]
    occupied = set().union(*model["pieces"])
    reflected = {(x, 2 * axis - y) for x, y in occupied if y != axis}
    return {b for b in reflected if inside_board({b}) and b not in occupied}


def render(model, frame):
    out = [list(row) for row in frame]
    reflected = update_reflection(model)
    owner = {b: i for i, piece in enumerate(model["pieces"]) for b in piece}
    axis_selected = model["selected"] is None
    for by in range(N):
        for bx in range(N):
            b = (bx, by)
            # Each layer supplies a ring, an optional solid centre, and a hole fill.
            layers = []
            if b in owner:
                layers.append((PIECE, None, DOT if owner[b] == model["selected"] else None))
            if b in reflected:
                layers.append((REFL, None, REFL))
            if b in model["targets"]:
                layers.append((TGT, TGT, None))
            if by == model["axis"]:
                layers.append((WALL, DOT if axis_selected else None, None))
            ring = layers[0][0] if layers else BG
            centre = next((l[1] for l in layers if l[1] is not None), None)
            if centre is None:
                centre = next((l[2] for l in layers if l[2] is not None), BG)
            for j in range(3):
                for i in range(3):
                    out[3 * by + j][3 * bx + i] = centre if (i, j) == (1, 1) else ring
    return out


def update_counter(out, changed):
    if changed:
        for y in range(63):
            if out[y][63] != HUD_ON:
                out[y][63] = HUD_ON
                break


def transition_function(state, action, frame):
    aid = action["action_id"] if isinstance(action, dict) else action
    model = parse(frame)
    changed = False
    if aid == 5:
        changed = update_selection(model)
    elif aid in (1, 2, 3, 4):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[aid]
        if model["selected"] is None:
            changed = update_wall(model, dx, dy)
        else:
            changed = update_player(model, dx, dy)
    out = render(model, frame)
    update_counter(out, changed)
    return out
