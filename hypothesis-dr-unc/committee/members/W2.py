import math

def predict(config):
    g = config.get
    bs = g("batch_size", 1024) or 1024
    ep = g("epochs", 10)
    steps = g("train_steps", None)
    sub = g("train_subset_size", 50000)
    if steps is None:
        steps = ep * sub / bs
    else:
        ep = steps * bs / sub
    aug = str(g("augmentation", ""))
    ls = g("label_smoothing", 0.1)
    cut = "cutout" in aug
    t = 0.8 + steps * (0.0165 + (0.0004 if cut else 0.0))
    t *= (bs / 1024.0) ** 0.0 if bs else 1.0
    if not g("torch_compile", True):
        t *= 1.2
    # accuracy: saturating in epochs
    acc = 0.7435 + 0.0087 * (ep - 10) if ep >= 8 else 0.7435 - 0.02 * (10 - ep) ** 1.0 * 0.5
    if cut:
        acc -= 0.006
    else:
        acc += 0.006
    if abs(ls - 0.1) < abs(ls - 0.2):
        acc += 0.006
    if "random_flip" in aug:
        acc -= 0.004
    if "translate" not in aug:
        acc -= 0.02
    acc = min(acc, 0.95)
    return {"time": t, "accuracy": acc}
