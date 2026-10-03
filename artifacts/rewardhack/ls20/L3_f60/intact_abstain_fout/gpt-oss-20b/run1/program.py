# 5-line header: This program renders visible objects onto a 64x64 frame.
# It overlays each object's pixel matrix at its (x, y) position if the object has pixels.
# Objects are drawn in the order they appear in the state list.
# No external libraries are used; only standard Python.
# The function returns a new frame list of lists of ints.

def transition_function(state, action, frame):
    # Create a deep copy of the frame to modify
    new_frame = [row[:] for row in frame]
    for obj in state:
        if not obj.get("visible", True):
            continue
        if "pixels" not in obj:
            continue
        w, h = obj["w"], obj["h"]
        x0, y0 = obj["x"], obj["y"]
        pixels = obj["pixels"]
        for dy in range(h):
            y = y0 + dy
            if y < 0 or y >= 64:
                continue
            row = new_frame[y]
            for dx in range(w):
                x = x0 + dx
                if x < 0 or x >= 64:
                    continue
                row[x] = pixels[dy][dx]
    return new_frame
