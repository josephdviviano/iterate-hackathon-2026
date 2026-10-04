import math

def predict(config):
    g = config.get
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 384.0, 576.0
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    steps = float(steps)
    per = 0.00522 + 1.91e-7 * c1 * c1 + 4.8e-8 * c2 * c2 + 8.6e-9 * c3 * c3
    per *= (bs / 1024.0) ** 0.8
    f = g("low_res_step_fraction", 0) or 0
    f = float(f)
    ph = g("num_resolution_phases", 0) or 0
    if ph == 3:
        k = 0.5
    elif ph == 2:
        k = 0.21
    else:
        k = 0.33
    t = steps * per * (1 - k * f)
    acc = 0.7549
    aug = str(g("augmentation", "alternating_flip+translate4"))
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    if "translate2" in aug:
        acc += 0.003
    ls = float(g("label_smoothing", 0.1) or 0)
    acc -= 0.03 * abs(ls - 0.1) if ls > 0.1 else 0.0
    if ls == 0:
        acc -= 0.003
    if ep <= 10:
        acc += 0.0806 * math.log(ep / 10.0)
    else:
        acc += 0.004 * (ep - 10)
    acc += 0.0146 * math.log(c2 / 384.0) + 0.0217 * math.log(c3 / 576.0) + 0.018 * math.log(c1 / 128.0)
    acc -= 0.011 * f
    return {"time": t, "accuracy": acc}
