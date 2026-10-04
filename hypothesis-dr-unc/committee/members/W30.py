import math

def predict(config):
    g = config.get
    try:
        c1, c2, c3 = [float(x) for x in str(g("conv_channels", "128,384,576")).split(",")[:3]]
    except Exception:
        c1, c2, c3 = 128., 384., 576.
    bs = float(g("batch_size", 1024) or 1024)
    ep = g("epochs", None)
    steps = g("train_steps", None)
    if ep is None:
        ep = float(steps) * bs / 50000. if steps else 10.
    ep = float(ep)
    if not steps:
        steps = ep * 50000. / bs
    steps = float(steps)
    f = (c1*c2*1024 + c2*c2*256 + c2*c3*256 + c3*c3*64) / 1e6
    fr = float(g("low_res_step_fraction", 0) or 0)
    ph = g("num_resolution_phases", 2) or 2
    if fr > 0:
        if ph >= 3:
            rf = (1 + 0.766 + 0.5625) / 3.
        else:
            rf = fr * 0.5625 + (1 - fr)
        ov = 0.13 * (f / 109.) ** 3
    else:
        rf = 1.0
        ov = 0.0
    opt = str(g("optimizer", ""))
    per_ep = 0.106 + 0.0033 * f + 0.00125 * c1
    t = per_ep * ep * rf + 0.0015 * steps + ov
    if "muon" in opt:
        t += 0.67
    aug = str(g("augmentation", ""))
    acc = 0.755 + 0.005 * (ep - 10) + 0.02 * math.log(f / 165.8)
    acc += 0.015 * math.log(c1 / 128.)
    if fr > 0:
        acc -= 0.005 if ph >= 3 else 0.0065
    if "cutout" in aug:
        acc -= 0.012
    if "random_flip" in aug:
        acc -= 0.004
    if "translate2" in aug:
        acc += 0.003
    ls = g("label_smoothing", 0.1)
    ls = 0.1 if ls is None else float(ls)
    if ls < 0.05:
        acc -= 0.003
    if ls > 0.15:
        acc -= 0.004
    if bs > 1024:
        acc -= 0.0000008 * (bs - 1024)
    if "muon" in opt:
        acc += 0.008
    return {"time": t, "accuracy": acc}
