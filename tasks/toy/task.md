# Task: toy

A tiny stand-in problem for testing research frameworks. `train.py` reads the hyperparameters in
`model/params.py` and prints a `loss` and a `time`. The goal: the lowest `time` with `loss <= 0.6`.
Only `model/` may be edited. A width above 64 crashes (simulated out-of-memory).
