"""A committee that predicts the outcome of a training-recipe idea before the sweep is run.

Members are seeded roles that read the recipe, the directives and every prior
finding except the one being predicted, and return P(the idea improves the
recipe at matched cost), a predicted accuracy delta and a predicted time delta.
Disagreement among members marks the ideas worth a sweep. Scored against the
measured sweep outcomes, so the filter's calibration and the sweeps it would
have saved are numbers, not claims.
"""
