"""Toy task: a stand-in "training run" that is instant and deterministic.

loss falls with better hyperparameters, time grows with width x epochs. Crashes if width > 64.
"""
import hashlib
import math
import pathlib

from model.params import EPOCHS, LR, WIDTH

src = pathlib.Path("model/params.py").read_text()
noise = int(hashlib.md5(src.encode()).hexdigest(), 16) % 1000 / 1e5
if WIDTH > 64:
    raise MemoryError("width too large (simulated OOM)")
loss = (math.log10(LR) + 1.5) ** 2 + 4.0 / WIDTH + 2.0 / EPOCHS + noise
print("training ...")
print(f"loss: {loss:.5f}")
print(f"time: {WIDTH * EPOCHS * 0.01:.4f}")
