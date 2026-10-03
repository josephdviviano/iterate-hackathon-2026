# Transition rule: The game is a simple grid world where objects are
# rendered as solid rectangles.  The player is a 5×5 block of colour 3.
# Yellow rings are colour 9, diamonds are colour 11 (hex 'b'), recolour
# tokens are colour 14 (hex 'e'), and budget counters (step bar) are
# colour 11.  The step bar is drawn at y=61 with its current width.
# All other objects are drawn at their (x, y) positions with their
# width and height.  No other dynamic behaviour is inferred.

def transition_function(state, action, frame):
    # Create a fresh 64×64 frame filled with 0
    new_frame = [[0] * 64 for _ in range(64)]

    # Helper to draw a rectangle of a given colour
    def draw_rect(x, y, w, h, colour):
        for dy in range(h):
            yy = y + dy
            if 0 <= yy < 64:
                row = new_frame[yy]
                for dx in range(w):
                    xx = x + dx
                    if 0 <= xx < 64:
                        row[xx] = colour

    # Map object types to colours (int values)
    colour_map = {
        'player': 3,
        'refuel': 9,
        'target': 11,   # diamond or legend
        'button': 14,
        'counter': 11,  # budget counter
    }

    # Render all objects
    for obj in state:
        t = obj.get('type')
        colour = colour_map.get(t, 0)
        # Special handling for step bar: draw at y=61
        if t == 'counter' and 'budget' in obj.get('tags', []):
            y = 61
            x = obj.get('x', 0)
            w = obj.get('w', 0)
            h = obj.get('h', 1)
            draw_rect(x, y, w, h, colour)
        else:
            x = obj.get('x', 0)
            y = obj.get('y', 0)
            w = obj.get('w', 0)
            h = obj.get('h', 0)
            draw_rect(x, y, w, h, colour)

    return new_frame
