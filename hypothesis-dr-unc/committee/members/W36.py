import math

def _f(c, k, d):
    try:
        v = c.get(k, d)
        return d if v is None or v == "" else float(v)
    except Exception:
        return d

def predict(config):
    c = config
    ch = str(c.get("conv_channels", "128,256,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 256.0, 576.0
    bs = max(_f(c, "batch_size", 1024), 1.0)
    ep = _f(c, "epochs", 11.0)
    if c.get("train_steps"):
        ep = _f(c, "train_steps", 537) * bs / 50000.0
    f = _f(c, "low_res_step_fraction", 0.0)
    ls = _f(c, "label_smoothing", 0.1)
    aug = str(c.get("augmentation", "alternating_flip+translate2"))
    opt = str(c.get("optimizer", "sgd_lookahead_ema"))

    # time: samples * per-sample cost (affine in widths), batch-size efficiency, low-res savings
    u = -3.9 + 0.042 * c1 + 0.0305 * c2 + 0.0081 * c3
    u = max(u, 3.0)
    bf = (1 + 150.0 / bs) / (1 + 150.0 / 1024.0)
    t = ep * 50000.0 * u / 1e6 * bf * (1 - 0.35 * f)
    steps = ep * 50000.0 / bs
    if "muon" in opt:
        if "muon_conv" in opt:
            ms = 1.65
        elif "+muon" in opt:
            ms = 2.3
        else:
            ms = 1.5
        t += steps * ms / 1000.0

    # accuracy (percent)
    a = 75.67 + 0.5 * (ep - 11.0)
    a += 0.5 * (c1 - 128.0) / 32.0 if c1 < 128 else 0.1 * (c1 - 128.0) / 32.0
    a += 0.7 * (c2 - 256.0) / 128.0
    a += 0.8 * (c3 - 576.0) / 192.0 if c3 < 576 else 0.0023 * (c3 - 576.0)
    if "translate4" in aug:
        a -= 0.4
    if "cutout" in aug:
        a -= 0.8
    if "random_flip" in aug:
        a -= 0.4
    a -= 3.0 * abs(ls - 0.1)
    if f > 0:
        a -= 1.1 * f if f <= 0.5 else 0.55 + 5.4 * (f - 0.5)
    if "muon_conv" in opt:
        a += 1.1
    elif "muon" in opt:
        a += 0.8
    if bs > 1600:
        a -= 0.15
    if bs < 900:
        a -= 0.05
    a = min(a, 95.0)
    return {"time": t, "accuracy": a / 100.0}
