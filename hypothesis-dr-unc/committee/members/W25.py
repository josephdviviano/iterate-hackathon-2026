import math

def predict(config):
    g = config.get
    def num(k, d):
        try:
            v = g(k, d)
            return d if v in (None, "") else float(v)
        except Exception:
            return d
    ch = str(g("conv_channels", "128,256,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 256.0, 576.0
    bs = num("batch_size", 1024.0)
    ep = num("epochs", 11.0)
    steps = ep * 50000.0 / bs
    f = (1024 * c1 * c1 + 256 * (c1 * c2 + c2 * c2) + 64 * (c2 * c3 + c3 * c3)) / 1e6
    cost = (0.48 + 0.0103 * f) * 1e-5
    frac = num("low_res_step_fraction", 0.0)
    ph = num("num_resolution_phases", 2.0 if frac > 0 else 1.0)
    if frac > 0:
        if ph >= 3:
            rf = 1 - frac * 0.4375 - frac * 0.234
        else:
            rf = 1 - frac * 0.4375
    else:
        rf = 1.0
    aug = str(g("augmentation", ""))
    opt = str(g("optimizer", ""))
    t = 0.3 + 0.0012 * steps + steps * bs * cost * rf
    if "cutout" in aug:
        t += 0.3
    if "muon" in opt:
        t += 0.8
    ls = num("label_smoothing", 0.1)
    acc = 0.7567 + 0.0035 * (ep - 11) + 0.02 * math.log(f / 72.6)
    acc -= 0.012 * frac * (0.9 if ph >= 3 else 1.0) if frac > 0 else 0
    if "cutout" in aug:
        acc -= 0.008
    if "translate4" in aug:
        acc -= 0.003
    if "random_flip" in aug:
        acc -= 0.004
    if ls < 0.05:
        acc -= 0.003
    elif ls > 0.15:
        acc -= 0.003
    if "muon" in opt:
        acc += 0.0075
    if bs < 900:
        acc -= 0.0004
    return {"time": t, "accuracy": acc}
