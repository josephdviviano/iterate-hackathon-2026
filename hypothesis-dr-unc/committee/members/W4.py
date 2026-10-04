import math

def predict(config):
    g = config.get
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * float(g("train_subset_size", 50000)) / bs
    steps = float(steps)
    aug = str(g("augmentation", ""))
    ls = float(g("label_smoothing", 0.1) or 0.0)
    res = float(g("input_resolution", 32) or 32)
    t = 0.6 + 0.0172 * steps * (bs / 1024.0) ** 0.9 * (res / 32.0) ** 2
    if "cutout" in aug:
        t += 0.12
    if g("torch_compile", True) is False:
        t *= 1.3
    A = 0.8177 + 0.0
    acc = A - 0.742 / max(ep, 1.0)
    if "cutout" not in aug:
        acc += 0.007
    if ls > 0.15:
        acc -= 0.004
    elif ls > 0.0:
        acc += 0.004
    if "random_flip" in aug and "alternating" not in aug:
        acc -= 0.004
    if "translate" not in aug:
        acc -= 0.02
    if not g("ema", True):
        acc -= 0.003
    acc = max(0.01, min(acc, 0.95))
    return {"time": t, "accuracy": acc}
