"""Structure: (fill in)
"""
from fractions import Fraction


def strategy(n, m, t):
    # Bernoulli(m): atoms {0, 1}. Replace with a better law.
    return [0, 1], [1 - m, m]


def confidence(n, m, t):
    # Probability that strategy(n, m, t) is within 1e-4 of the supremum. Replace.
    return 0.0
