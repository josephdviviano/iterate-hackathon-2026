# Mechanics:
# - Action 1: move player up 5 cells.
# - Action 2: move player down 5 cells.
# - Action 3: move player left 5 cells.
# - Action 4: move player right 5 cells.
# - Player can only move onto cells coloured '3' (the path) in the current frame.
# - Player sprite: 5×5 block, top two rows colour 'c' (12), bottom three rows colour '9'.
# - The step_bar counter is drawn on rows 61 and 62 using colour 'b' (11) for its
#   active segment (width w starting at x). Before drawing the new segment we
#   clear the previous segment cells (those currently coloured 'b') on those rows.

def transition_function(state, action, frame):
    # ----- helpers -----
    def find_obj(t, name=None):
        for o in state:
            if o.get("type") == t:
                if name is None or o.get("name") == name:
                    return o
        return None

    # Detect whether the frame is a list of strings or a list of int lists.
    # If the first row is a string, we treat the whole frame as strings.
    is_str = isinstance(frame[0], str)

    # Convert to mutable rows.
    if is_str:
        new_frame = [list(row) for row in frame]          # list of chars
        path_c = '3'
        player_top = 'c'
        player_bot = '9'
        bar_c = 'b'
    else:
        new_frame = [list(row) for row in frame]          # list of ints
        path_c = 3
        player_top = 12
        player_bot = 9
        bar_c = 11

    # ----- player movement -----
    player = find_obj("player")
    if player:
        px, py = player["x"], player["y"]
        w, h = player["w"], player["h"]

        dx = dy = 0
        if action == 1:      # up
            dy = -5
        elif action == 2:    # down
            dy = 5
        elif action == 3:    # left
            dx = -5
        elif action == 4:    # right
            dx = 5

        nx, ny = px + dx, py + dy

        # bounds check
        if 0 <= nx <= 64 - w and 0 <= ny <= 64 - h:
            # verify destination cells are all path colour
            can_move = True
            for yy in range(ny, ny + h):
                for xx in range(nx, nx + w):
                    if new_frame[yy][xx] != path_c:
                        can_move = False
                        break
                if not can_move:
                    break
            if can_move:
                # erase old player (restore path colour)
                for yy in range(py, py + h):
                    for xx in range(px, px + w):
                        new_frame[yy][xx] = path_c
                # draw player at new location
                for dy_i in range(h):
                    colour = player_top if dy_i < 2 else player_bot
                    for dx_i in range(w):
                        new_frame[ny + dy_i][nx + dx_i] = colour

    # ----- step bar rendering -----
    step_bar = find_obj("counter", "step_bar")
    if step_bar:
        sb_x = step_bar["x"]
        sb_w = step_bar["w"]
        bar_rows = (61, 62)

        # Clear previous bar cells (those currently coloured bar_c)
        for r in bar_rows:
            for c in range(64):
                if new_frame[r][c] == bar_c:
                    new_frame[r][c] = path_c

        # Draw new bar segment
        for r in bar_rows:
            for c in range(sb_x, sb_x + sb_w):
                if 0 <= c < 64:
                    new_frame[r][c] = bar_c

    # Convert back to original representation
    if is_str:
        return [''.join(row) for row in new_frame]
    else:
        return [list(row) for row in new_frame]
