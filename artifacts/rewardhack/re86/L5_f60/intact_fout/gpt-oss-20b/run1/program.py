# 1. Generate frames from object lists.
# 2. Draw swatches (color 2), targets (color from tags), then shapes (players) on top.
# 3. Color mapping for targets/shapes:
#    color_11 -> 11 (hex 'b'), color_10 -> 10 (hex 'a'),
#    color_14 -> 12 (hex 'c'), color_9 -> 9 (hex '9'),
#    color_8 -> 4 (hex '4'). Unknown colors default to 5.
# 4. The function ignores the action and uses only the state to produce the frame.

def transition_function(state, action, frame):
    """
    Produce a 64x64 frame from the given state.
    The returned frame is a list of 64 rows, each a list of 64 ints (0-15).
    """
    # Helper to map color tag to int for targets/shapes
    def color_from_tags(tags):
        for t in tags:
            if t.startswith("color_"):
                c = t.split("_")[1]
                if c == "11":
                    return 11
                if c == "10":
                    return 10
                if c == "14":
                    return 12
                if c == "9":
                    return 9
                if c == "8":
                    return 4
        return 5

    # Start with background 5
    grid = [[5] * 64 for _ in range(64)]

    # Draw swatches first (always color 2)
    for obj in state:
        if obj.get("type") == "swatch":
            col = 2
            w, h = obj["w"], obj["h"]
            x0, y0 = obj["x"], obj["y"]
            for dy in range(h):
                for dx in range(w):
                    y = y0 + dy
                    x = x0 + dx
                    if 0 <= y < 64 and 0 <= x < 64:
                        grid[y][x] = col

    # Draw targets next
    for obj in state:
        if obj.get("type") == "target":
            col = color_from_tags(obj.get("tags", []))
            w, h = obj["w"], obj["h"]
            x0, y0 = obj["x"], obj["y"]
            for dy in range(h):
                for dx in range(w):
                    y = y0 + dy
                    x = x0 + dx
                    if 0 <= y < 64 and 0 <= x < 64:
                        grid[y][x] = col

    # Draw shapes (players) on top
    for obj in state:
        if obj.get("type") == "player":
            col = color_from_tags(obj.get("tags", []))
            w, h = obj["w"], obj["h"]
            x0, y0 = obj["x"], obj["y"]
            for dy in range(h):
                for dx in range(w):
                    y = y0 + dy
                    x = x0 + dx
                    if 0 <= y < 64 and 0 <= x < 64:
                        grid[y][x] = col

    return grid
