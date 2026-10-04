import math

def _f(c, k, d):
    v = c.get(k, d)
    try:
        if v is None or v == "":
            return d
        return float(v)
    except Exception:
        return d

def predict(config):
    c = config
    ch = str(c.get("conv_channels", "128,256,576"))
    try:
        a = [float(x) for x in ch.split(",")]
        c1, c2, c3 = a[0], a[1], a[2]
    except Exception:
        c1, c2, c3 = 128.0, 256.0, 576.0
    bs = max(1.0, _f(c, "batch_size", 1024.0))
    ep = _f(c, "epochs", 0.0)
    if ep <= 0:
        st = _f(c, "train_steps", 537.0)
        ep = st * bs / 50000.0
    steps = ep * 50000.0 / bs
    f = _f(c, "low_res_step_fraction", 0.0)
    phases = _f(c, "num_resolution_phases", 0.0)
    if f > 0 and phases < 2:
        phases = 2
    lo = _f(c, "low_res_input_resolution", 24.0)
    mid = _f(c, "mid_res_input_resolution", 28.0)
    ls = _f(c, "label_smoothing", 0.1)
    aug = str(c.get("augmentation", "alternating_flip+translate2"))
    opt = str(c.get("optimizer", "sgd_lookahead_ema"))

    # conv flops (x9 omitted, millions) over three blocks with pooling
    F = (24 * c1 * 1024 + 2 * c1 * c1 * 256
         + c1 * c2 * 256 + 2 * c2 * c2 * 64
         + c2 * c3 * 64 + 2 * c3 * c3 * 16) / 1e6
    p1024 = 0.00203 + 0.000205 * F
    # lower resolution phases shrink per-sample work with pixel count
    s = 1.0 - f * (1.0 - (lo / 32.0) ** 2)
    extra = 0.0
    if phases >= 3:
        s -= f * (1.0 - (mid / 32.0) ** 2)
        extra = 1.2 * (phases - 2)  # recompile / graph switch per extra shape
    step = 0.0022 + p1024 * (bs / 1024.0) * s
    if "muon_conv" in opt:
        step += 0.0015
    elif "muon_sgd" in opt:
        step += 0.0013
    elif "muon" in opt:
        step += 0.0020
    t = 0.1 + extra + steps * step

    acc = 0.7567
    d = ep - 11.0
    acc += 0.0052 * d if d < 0 else 0.001 * d
    g2 = 0.012 * math.log2(c2 / 256.0)
    if g2 > 0:
        g2 /= 1.0 + 0.5 * max(0.0, ep - 10.0)
    acc += g2
    l3 = math.log2(c3 / 576.0)
    acc += 0.0195 * l3 if l3 < 0 else 0.0195 * 0.7 * l3
    l1 = math.log2(c1 / 128.0)
    acc += 0.0125 * l1 if l1 < 0 else 0.0062 * l1
    if f > 0:
        rf = (32.0 - lo) / 8.0
        acc -= 0.006 * (f / 0.5) ** 3 * rf
        if phases >= 3:
            acc -= 0.0035
    acc -= 0.001 * max(0.0, math.log2(bs / 1024.0))
    if "muon_conv" in opt:
        acc += 0.011
    elif "muon" in opt:
        acc += 0.009
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    if "translate4" in aug:
        acc -= 0.0035
    if ls >= 0.15:
        acc -= 0.004
    elif ls < 0.05:
        acc -= 0.003
    return {"time": t, "accuracy": acc}
