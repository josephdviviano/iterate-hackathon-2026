import math

def _f(v, d):
    try:
        if v is None or v == "":
            return d
        return float(v)
    except Exception:
        return d

def predict(config):
    c = config
    ch = str(c.get("conv_channels", "128,384,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 384.0, 576.0
    bs = _f(c.get("batch_size"), 1024.0)
    ep = _f(c.get("epochs"), 10.0)
    steps = _f(c.get("train_steps"), 0.0)
    if steps <= 0:
        steps = ep * 50000.0 / bs
    aug = str(c.get("augmentation", "alternating_flip+translate2"))
    ls = _f(c.get("label_smoothing"), 0.1)
    frac = _f(c.get("low_res_step_fraction"), 0.0)
    ph = _f(c.get("num_resolution_phases"), 2.0)
    low = frac
    mid = frac if (ph >= 3 and frac > 0) else 0.0
    cut = "cutout" in aug
    rflip = "random_flip" in aug
    t = 0.0
    if "translate" in aug:
        try:
            t = float(aug.split("translate")[1].split("+")[0])
        except Exception:
            t = 2.0

    # time: per-step = fixed + conv FLOPs term, scaled sublinearly by batch
    F = (c1 * c1 * 1024 + (c1 * c2 + c2 * c2) * 256 + (c2 * c3 + c3 * c3) * 64) / 1e6
    cost = 0.0060 + 1.2e-4 * F
    cost *= (bs / 1024.0) ** 0.85
    if cut:
        cost *= 1.02
    lr = 1.0 - 0.73 * (low * 0.4375 + mid * 0.234)
    time = steps * cost * lr

    # accuracy
    acc = 0.755
    acc += 0.05 * math.log(max(ep, 1.0) / 10.0)
    acc -= 0.0165 * math.log(384.0 / c2) + 0.02 * math.log(576.0 / c3) \
        + 0.018 * math.log(128.0 / c1)
    acc += 0.0018 * (4.0 - max(t, 2.0)) if t < 4 else 0.0
    if t > 4:
        acc -= 0.002 * (t - 4)
    acc -= 0.3 * (ls - 0.1) ** 2
    if cut:
        acc -= 0.0095
    if rflip:
        acc -= 0.004
    acc -= 0.0105 * low + 0.005 * mid
    return {"time": time, "accuracy": acc}
