# Mechanics: none modelled yet. This is a baseline that predicts no change (returns a copy of the before state).
# Stopped on request before the workspace was downloaded and before check.py was run.
# Open hypothesis: under ACTION3, counter objects change tags,w,x in some transitions and stay unchanged in others.
# Hypothesis to test first: a field of the counter itself (tags/pixels/visible/layer) decides which happens.
# Hidden state: none (stateless).
import copy


def transition_function(state, action):
    return copy.deepcopy(state)
