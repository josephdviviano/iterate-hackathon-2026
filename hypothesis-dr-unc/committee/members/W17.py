import math

def predict(config):
    g = config.get
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 384.0, 576.0
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    steps = float(steps)
    ms = 1.68 + 0.031 * c2 + 0.0082 * c3 + 0.0 * c1
    ms *= 0.15 + 0.85 * bs / 1024.0
    rm = g("resize_mode", "")
    frac = float(g("low_res_step_fraction", 0.5))
    res = float(g("low_res_input_resolution", 24))
    if rm:
        r = (res / 32.0) ** 2
        ms *= 1 - frac * (1 - r) * 0.73
    t = steps * ms / 1000.0

    acc = 0.7585
    if ep < 10:
        acc -= 0.008 * (10 - ep)
    else:
        acc += 0.003 * (ep - 10)
    acc -= 0.00004 * (384 - c2) + 0.00007 * (576 - c3)
    aug = str(g("augmentation", ""))
    if "translate4" in aug:
        acc -= 0.003
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    ls = float(g("label_smoothing", 0.1))
    acc -= 0.03 * abs(ls - 0.1) if ls > 0.1 else 0.03 * (0.1 - ls)
    if rm:
        acc -= 0.005
    acc -= 0.0000008 * (bs - 1024) * -1 if bs < 1024 else 0
    return {"time": t, "accuracy": acc}
