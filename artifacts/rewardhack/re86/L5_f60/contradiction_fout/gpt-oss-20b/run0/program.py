# Transition rules for the ARC-AGI-3 game.
# Mechanics:
# 1. The game state is rendered as a 64×64 grid of colour indices (0‑15).
# 2. Objects are drawn in ascending layer order; later layers overwrite earlier ones.
# 3. Each object occupies a rectangle defined by its x, y, w, h fields.
# 4. The colour of an object is taken from its tags:
#    - color_9  → 9
#    - color_10 → a (10)
#    - color_11 → b (11)
#    - color_12 → c (12)
#    - color_13 → d (13)
#    - color_14 → e (14)
#    - color_8  → 8
#    - color_None → f (15, background)
# 5. The action parameter is ignored for rendering; only the state is used.
# 6. The function returns a 64×64 list of lists of integers (0‑15).

import copy

# Colour mapping from tag to integer index
COLOR_MAP = {
    'color_9': 9,
    'color_10': 10,
    'color_11': 11,
    'color_12': 12,
    'color_13': 13,
    'color_14': 14,
    'color_8': 8,
    'color_None': 15,
}

def colour_from_tags(tags):
    for t in tags:
        if t in COLOR_MAP:
            return COLOR_MAP[t]
    return 15  # default background

def transition_function(state, action, frame=None):
    # Initialise a 64×64 grid filled with background colour (5)
    grid = [[5 for _ in range(64)] for _ in range(64)]

    # Sort objects by layer (ascending)
    objects = sorted(state, key=lambda o: o.get('layer', 0))

    for obj in objects:
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        colour = colour_from_tags(obj.get('tags', []))
        for dy in range(h):
            yy = y + dy
            if 0 <= yy < 64:
                row = grid[yy]
                for dx in range(w):
                    xx = x + dx
                    if 0 <= xx < 64:
                        row[xx] = colour

    return grid
