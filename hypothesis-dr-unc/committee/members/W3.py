import math

def predict(config):
    c = config
    bs = float(c.get("batch_size", 1024))
    ep = float(c.get("epochs", 10))
    sub = float(c.get("train_subset_size", 50000))
    steps = c.get("train_steps")
    if not steps:
        steps = ep * sub / bs
    steps = float(steps)
    res = float(c.get("input_resolution", 32))
    params = float(c.get("num_params_approx", 2500000))
    cost = (bs / 1024.0) ** 0.9 * (res / 32.0) ** 2 * (params / 2.5e6) ** 0.8
    if not c.get("torch_compile", True):
        cost *= 1.3
    t = 0.8 + 0.0165 * steps * cost
    aug = str(c.get("augmentation", "alternating_flip+translate4"))
    ls = float(c.get("label_smoothing", 0.1))
    eff_ep = steps * bs / 50000.0
    acc = 0.7304 + 0.0806 * math.log(max(eff_ep, 1.0) / 8.5)
    # augmentation / regularisation interactions: short runs prefer little regularisation
    if "cutout" in aug:
        acc -= 0.008 * (1.0 if eff_ep < 12 else 0.5)
    else:
        acc += 0.0
    acc += 0.008 if "cutout" in aug else 0.0
    acc += 0.0115 if "cutout" not in aug else 0.0
    acc += -0.35 * (ls - 0.2) * 0.1 if False else 0.035 * (0.2 - ls)
    if "alternating" not in aug and "flip" in aug:
        acc -= 0.004
    if "flip" not in aug:
        acc -= 0.01
    if not c.get("ema", True):
        acc -= 0.003
    if not c.get("whitening_init", True):
        acc -= 0.01
    acc = min(acc, 0.97)
    return {"time": t, "accuracy": acc}
