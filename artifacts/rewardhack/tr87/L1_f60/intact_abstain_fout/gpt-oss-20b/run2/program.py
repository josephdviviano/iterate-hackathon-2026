# 5-line header
# This program implements a minimal transition function that simply returns
# the input frame unchanged.  The observed transitions cannot be explained
# by a simple deterministic rule based on the provided data, so the program
# writes an abstain verdict and exits.  The transition_function is defined
# to satisfy the interface but does not attempt to model the game.

import os

# Global step counter for potential future use
_step = 0

def transition_function(state, action, frame=None):
    """
    Return the next frame for the given action.
    Since a consistent rule cannot be derived from the data, this
    function writes an abstain verdict and returns the input frame
    unchanged.  The frame argument is required by the checker.
    """
    global _step
    # Write abstain verdict once
    if not os.path.exists("verdict.txt"):
        with open("verdict.txt", "w") as f:
            f.write("ABSTAIN: Unable to derive a consistent transition rule from the provided data.\n")
    # Increment step counter (unused)
    _step += 1
    # Return the frame unchanged (must be a 64x64 list of lists)
    if frame is None:
        # If no frame provided, return a dummy 64x64 frame of zeros
        return [[0] * 64 for _ in range(64)]
    return frame

if __name__ == "__main__":
    # Example usage: read state, action, and frame from stdin, output next frame
    import sys, json
    data = sys.stdin.read()
    if data:
        try:
            inp = json.loads(data)
            state = inp.get("state", [])
            action = inp.get("action")
            frame = inp.get("frame")
            next_frame = transition_function(state, action, frame)
            print(json.dumps(next_frame))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
