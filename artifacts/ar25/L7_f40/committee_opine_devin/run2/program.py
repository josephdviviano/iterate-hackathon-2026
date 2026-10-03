# Mechanics: 63x63 board of 21x21 3x3 blocks; H axis = wall row (A1/A2), V axis = wall column (A3/A4).
# Selected entity (axis or piece) shows 0 centres; A5 cycles H -> V -> pieces (by min block, row-major) -> H.
# Pieces (ring 5) move in 4 dirs, blocked by board edge / other pieces; axes stop at the edge; blocked = no-op.
# Every piece block off an axis is mirrored across H, V and both (visible solid 4 with centre hole), skipping piece-covered dests.
# Layers piece > reflection > target > walls, but a target centre 11 shows through every hole/0 dot; HUD col 63 = one 12 per effective action. Unconfirmed: none.
B = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def blocks(frame):
    ring = [[frame[3 * r][3 * c] for c in range(B)] for r in range(B)]
    cen = [[frame[3 * r + 1][3 * c + 1] for c in range(B)] for r in range(B)]
    return ring, cen


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for dr, dc in DIRS.values():
                n = (r + dr, c + dc)
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def parse(frame):
    ring, cen = blocks(frame)
    walls = [(r, c) for r in range(B) for c in range(B) if ring[r][c] == 10]
    ah = max(range(B), key=lambda r: sum(1 for w in walls if w[0] == r))
    av = max(range(B), key=lambda c: sum(1 for w in walls if w[1] == c))
    targets = {(r, c) for r in range(B) for c in range(B) if ring[r][c] == 11 or cen[r][c] == 11}
    pcells = {(r, c) for r in range(B) for c in range(B) if ring[r][c] == 5}
    hv = sum(1 for r, c in walls if cen[r][c] == 0 and r == ah and c != av)
    vv = sum(1 for r, c in walls if cen[r][c] == 0 and c == av and r != ah)
    pieces, sel = [], None
    if hv > vv:
        sel = 'H'
    elif vv > hv:
        sel = 'V'
    else:
        seeds = [p for p in pcells if cen[p[0]][p[1]] == 0]
        if seeds:
            grown, st = set(seeds), list(seeds)
            while st:
                r, c = st.pop()
                for dr, dc in DIRS.values():
                    n = (r + dr, c + dc)
                    if n in pcells and n not in grown and cen[n[0]][n[1]] != 9:
                        grown.add(n)
                        st.append(n)
            pieces.append(grown)
            pcells = pcells - grown
            sel = 0
    pieces += comps(pcells)
    pieces.sort(key=min)
    if sel == 0:
        sel = next(i for i, p in enumerate(pieces) if p is grown)
    return {'ah': ah, 'av': av, 'sel': sel, 'pieces': pieces, 'targets': targets}


def on_board(r, c):
    return 0 <= r < B and 0 <= c < B


def step(m, action):
    """Returns the new model, or None when the action has no effect."""
    sel, n = m['sel'], len(m['pieces'])
    if action == 5:
        order = ['H', 'V'] + list(range(n))
        return dict(m, sel=order[(order.index(sel) + 1) % len(order)])
    if action not in DIRS:
        return None
    dr, dc = DIRS[action]
    if sel == 'H':
        nah = m['ah'] + dr
        return dict(m, ah=nah) if dr and on_board(nah, 0) else None
    if sel == 'V':
        nav = m['av'] + dc
        return dict(m, av=nav) if dc and on_board(0, nav) else None
    moved = {(r + dr, c + dc) for r, c in m['pieces'][sel]}
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != sel])
    if any(not on_board(*p) for p in moved) or moved & others:
        return None
    pieces = list(m['pieces'])
    pieces[sel] = moved
    order = sorted(range(n), key=lambda i: min(pieces[i]))
    return dict(m, pieces=[pieces[i] for i in order], sel=order.index(sel))


def reflections(m):
    ah, av = m['ah'], m['av']
    pc = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for r, c in pc:
        cand = []
        if r != ah:
            cand.append((2 * ah - r, c))
        if c != av:
            cand.append((r, 2 * av - c))
        if r != ah and c != av:
            cand.append((2 * ah - r, 2 * av - c))
        out |= {p for p in cand if on_board(*p) and p not in pc}
    return out


def render(m, frame, ticks):
    out = [row[:] for row in frame]
    ah, av, sel = m['ah'], m['av'], m['sel']
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m['pieces']):
        for cell in p:
            owner[cell] = i
    for r in range(B):
        for c in range(B):
            wall = r == ah or c == av
            dot = (sel == 'H' and r == ah) or (sel == 'V' and c == av)
            tgt = (r, c) in m['targets']
            under = 11 if tgt else (0 if wall and dot else 9)
            if (r, c) in owner:
                rc, cc = 5, (0 if owner[(r, c)] == sel and not tgt else under)
            elif (r, c) in refl:
                rc, cc = 4, (11 if tgt else 4)
            elif tgt:
                rc = cc = 11
            elif wall:
                rc, cc = 10, (0 if dot else 9)
            else:
                rc = cc = 9
            for dy in range(3):
                for dx in range(3):
                    out[3 * r + dy][3 * c + dx] = cc if dy == dx == 1 else rc
    for y in range(63):
        out[y][63] = 12 if y < ticks else 11
    return out


def transition_function(state, action, frame):
    aid = action.get('action_id') if isinstance(action, dict) else action
    m = parse(frame)
    nm = step(m, aid)
    if nm is None:
        return [row[:] for row in frame]
    ticks = sum(1 for y in range(63) if frame[y][63] == 12) + 1
    return render(nm, frame, ticks)
