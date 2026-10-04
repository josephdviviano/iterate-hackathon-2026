import math

def predict(config):
    g = config.get
    aug = str(g("augmentation", "alternating_flip+translate4"))
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    sub = float(g("train_subset_size", 50000) or 50000)
    if not steps:
        steps = ep * sub / bs
    steps = float(steps)
    ls = float(g("label_smoothing", 0.1) or 0.0)
    cut = "cutout" in aug
    t = steps * (0.01873 if cut else 0.01835)
    t *= (bs / 1024.0) ** 0.3 if bs != 1024 else 1.0
    if g("torch_compile", True) is False:
        t *= 1.4
    if not g("data_on_gpu", True):
        t += 2.0
    ep_eff = steps * bs / sub
    acc = 0.7435 + 0.0087 * (ep_eff - 10.0) if ep_eff < 10 else 0.7435 + 0.0087 * 1.0 * math.log(1 + (ep_eff - 10) * 3) / 3 * 3 * 0.5 if ep_eff > 10 else 0.7435
    if not cut:
        acc += 0.0105
    if "alternating" not in aug:
        acc -= 0.0045
    acc -= 0.01 * abs(ls - 0.1) * (1 if ls > 0.1 else 0.5) * 0.5
    if g("ema", True) is False:
        acc -= 0.004
    if g("whitening_init", True) is False:
        acc -= 0.006
    return {"time": t, "accuracy": min(acc, 0.9)}
