"""Structure: (fill in)
"""
from fractions import Fraction


def strategy(c):
    # Two atoms {0, 1}. Replace with a better law.
    return [0, 1], [Fraction(1, 2), Fraction(1, 2)]


def confidence(c):
    # Probability that strategy(c) is within 1e-4 of the supremum. Replace.
    return 0.0
