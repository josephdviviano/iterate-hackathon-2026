import math

def predict(config):
    g = config.get
    ch = str(g("conv_channels", "128,256,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 256.0, 576.0
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    steps = float(steps)
    ep_eff = steps * bs / 50000.0
    lf = float(g("low_res_step_fraction", 0) or 0)
    phases = g("num_resolution_phases", 0) or 0
    r = 0.48
    if lf > 0 and phases == 3:
        rf = 0.78
    elif lf > 0:
        rf = lf * r + (1 - lf)
    else:
        rf = 1.0
    F = (1024 * 9 * (24 * c1 + 2 * c1 * c1) + 256 * 9 * (c1 * c2 + 2 * c2 * c2)
         + 64 * 9 * (c2 * c3 + 2 * c3 * c3))
    opt = str(g("optimizer", "sgd_lookahead_ema"))
    mu = 0.0
    if "muon_conv" in opt:
        mu = 0.0012
    elif "muon" in opt:
        mu = 0.0016 if opt.startswith("sgd") else 0.001
    # per-step: fixed launch/optimizer cost + batch-proportional data/compute cost
    step = 0.002 + mu + (bs / 1024.0) * (0.0021 + 8.7e-12 * F * rf)
    t = step * steps

    aug = str(g("augmentation", "alternating_flip+translate2"))
    acc = 0.7567 + 0.06 * math.log(max(ep_eff, 1.0) / 11.0)
    acc += 0.018 * math.log(c1 / 128.0) + 0.017 * math.log(c2 / 256.0) \
        + 0.03 * math.log(c3 / 576.0)
    if lf <= 0.5:
        pen = 0.013 * lf
    else:
        pen = 0.0065 + 0.0544 * (lf - 0.5)
    if phases == 3:
        pen += 0.001
    acc -= pen
    if "cutout" in aug:
        acc -= 0.0085
    if "translate4" in aug:
        acc -= 0.0035
    if "random_flip" in aug:
        acc -= 0.004
    ls = float(g("label_smoothing", 0.1) or 0.0)
    acc -= 0.03 * abs(ls - 0.1)
    if "muon_conv" in opt:
        acc += 0.011
    elif "muon" in opt:
        acc += 0.008
    acc -= 0.0005 * math.log(bs / 1024.0, 2)
    return {"time": t, "accuracy": acc}
