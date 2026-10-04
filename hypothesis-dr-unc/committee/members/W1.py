import math

def predict(config):
    g = config.get
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    n = float(g("train_subset_size", 50000) or 50000)
    if not steps:
        steps = ep * n / bs
    steps = float(steps)
    res = float(g("input_resolution", 32) or 32)
    par = float(g("num_params_approx", 2500000) or 2500000)
    aug = str(g("augmentation", ""))
    ls = float(g("label_smoothing", 0.1) or 0.0)
    # time: fixed overhead (data to GPU, whitening, warm) + per-step cost
    step_cost = 0.0162 * (bs / 1024.0) * (res / 32.0) ** 2 * (par / 2.5e6) ** 0.9
    if g("torch_compile", True) is False:
        step_cost *= 1.3
    if g("mixed_precision", "bf16") in ("none", None, False):
        step_cost *= 1.6
    t = 1.05 + step_cost * steps
    if "cutout" in aug:
        t += 0.02
    # accuracy
    acc = 0.7435 + 0.081 * math.log(max(ep, 0.5) / 10.0)
    if "cutout" in aug:
        acc -= 0.0 
    else:
        acc += 0.008
    if "alternating" not in aug:
        acc -= 0.004
    if "translate" not in aug:
        acc -= 0.03
    acc += 0.04 * (0.2 - ls) * (-1) * 0.0
    acc += 0.004 if abs(ls - 0.1) < 0.06 and ls < 0.15 else 0.0
    if not g("ema", True):
        acc -= 0.003
    if not g("whitening_init", True):
        acc -= 0.005
    if n < 50000:
        acc -= 0.1 * (1 - n / 50000.0)
    acc = min(acc, 0.95)
    return {"time": t, "accuracy": acc}
