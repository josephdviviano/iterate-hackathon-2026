import math

def _f(x, d):
    try:
        if x is None or x == "":
            return d
        return float(x)
    except Exception:
        return d

def predict(config):
    c = config
    ch = str(c.get("conv_channels", "128,384,576"))
    try:
        w = [float(v) for v in ch.split(",")]
    except Exception:
        w = [128.0, 384.0, 576.0]
    while len(w) < 3:
        w.append(w[-1])
    w1, w2, w3 = w[0], w[1], w[2]
    bs = _f(c.get("batch_size"), 1024.0)
    ep = _f(c.get("epochs"), 0)
    if ep <= 0:
        ep = _f(c.get("train_steps"), 488) * bs / 50000.0
    samples = ep * 50000.0
    # conv MACs per sample (millions-ish units)
    macs = (1024 * (24 * w1 + 2 * w1 * w1) + 256 * (w1 * w2 + 2 * w2 * w2)
            + 64 * (w2 * w3 + 2 * w3 * w3)) / 1e6
    per = 3.5 + 0.075 * macs  # microseconds per sample
    f = _f(c.get("low_res_step_fraction"), 0.0)
    ph = int(_f(c.get("num_resolution_phases"), 2))
    r = _f(c.get("low_res_input_resolution"), 24.0)
    mr = _f(c.get("mid_res_input_resolution"), 28.0)
    if f > 0:
        red = f * (1 - (r / 32.0) ** 2)
        if ph >= 3:
            red += f * (1 - (mr / 32.0) ** 2)
        fac = 1 - red
    else:
        fac = 1.0
    aug = str(c.get("augmentation", ""))
    opt = str(c.get("optimizer", ""))
    t = 0.5 + samples * per * 1e-6 * fac
    if "cutout" in aug:
        t += 0.25
    if "muon" in opt:
        t += 0.8
    ls = _f(c.get("label_smoothing"), 0.1)
    acc = 0.755 + 0.045 * math.log(max(ep, 1.0) / 10.0)
    acc += 0.018 * math.log(w3 / 576.0) + 0.0 * 0
    acc += 0.016 * math.log(w2 / 384.0) * 0.9 + 0.017 * math.log(w1 / 128.0)
    if "translate2" in aug:
        acc += 0.003
    if "cutout" in aug:
        acc -= 0.012
    if "random_flip" in aug:
        acc -= 0.004
    if ls < 0.05:
        acc -= 0.003
    elif ls > 0.15:
        acc -= 0.002
    acc -= 0.012 * f
    if "muon" in opt:
        acc += 0.008
    return {"time": t, "accuracy": acc}
