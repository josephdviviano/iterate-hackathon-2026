"""Hoeffding's problem as a second task for the committee.

sup P(S_n <= t) over iid X_1..X_n in [0, 1] with E X = m, where
S_n = X_1 + ... + X_n and 0 <= t < n m. A candidate is a discrete measure.
Its value is certified in exact rational arithmetic, so a candidate is a
lower bound on the supremum or it is rejected.
"""
