import math

def _f(v, d):
    try:
        if v in (None, ""):
            return d
        return float(v)
    except Exception:
        return d

def predict(config):
    g = config.get
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        c = [float(x) for x in ch]
        while len(c) < 3:
            c.append(c[-1])
    except Exception:
        c = [128.0, 384.0, 576.0]
    F = (c[0]**2*1024 + c[1]**2*256 + c[2]**2*64) / 1e6
    # time-effective cost: the last stage (8x8 maps) is cheap per parameter
    Ft = (c[0]**2*1024 + c[1]**2*256 + 0.5*c[2]**2*64) / 1e6
    ep = _f(g("epochs", 10), 10.0)
    bs = _f(g("batch_size", 1024), 1024.0)
    steps = _f(g("train_steps", 0), 0.0)
    if steps <= 0:
        steps = ep * 50000 / bs
    eff_ep = steps * bs / 50000.0
    f = _f(g("low_res_step_fraction", 0), 0.0)
    lr = _f(g("low_res_input_resolution", 24), 24.0)
    phases = int(_f(g("num_resolution_phases", 2), 2))
    fac = 1.0 - f * (1 - (lr / 32.0) ** 2)
    k = 0.292 + 0.0091 * Ft
    t = 0.05 + ep * k * fac
    if f > 0:
        t += 0.3  # resize overhead
    t += 0.0015 * (steps - ep * 50000 / 1024.0)
    opt = str(g("optimizer", ""))
    muon = "muon" in opt
    if muon:
        t += 0.8
    aug = str(g("augmentation", ""))
    ls = _f(g("label_smoothing", 0.1), 0.1)
    pool = str(g("pooling", "global_max"))
    a = 0.7585 + 0.022 * math.log(F / 75.7)
    a += 0.007 * math.log(max(c[0], 16.0) / 128.0)
    a += 0.005 * (eff_ep - 10)
    if "translate4" in aug:
        a -= 0.0035
    if "random_flip" in aug:
        a -= 0.004
    if "cutout" in aug:
        a -= 0.012
    if ls > 0.15:
        a -= 0.003
    elif ls < 0.05:
        a -= 0.0025
    a -= 0.011 * f
    if phases >= 3 and f > 0:
        a -= 0.0015
    if "adaptive" in pool:
        a -= 0.002
    a -= 0.0008 * (bs / 1024.0 - 1) if bs > 1024 else 0.0
    if muon:
        a += 0.0078
    return {"time": t, "accuracy": a}
