import math

def predict(config):
    g = config.get
    try:
        c = [float(x) for x in str(g("conv_channels", "128,384,576") or "128,384,576").split(",")]
    except Exception:
        c = [128.0, 384.0, 576.0]
    while len(c) < 3:
        c.append(c[-1])
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = ep * 50000.0 / bs
    aug = str(g("augmentation", "alternating_flip+translate2"))
    ls = g("label_smoothing", 0.1)
    ls = 0.1 if ls is None else float(ls)
    f = float(g("low_res_step_fraction", 0) or 0)
    res = float(g("low_res_input_resolution", 24) or 24)
    mid = float(g("mid_res_input_resolution", 28) or 28)
    phases = float(g("num_resolution_phases", 0) or 0)
    # per-step ms: affine in widths, sublinear in batch size
    ms = -3.84 + 0.043 * c[0] + 0.0311 * c[1] + 0.0082 * c[2]
    ms = max(ms, 2.0) * (bs / 1024.0) ** 0.83
    # low-res phases cut compute roughly by pixel count
    r_lo = (res / 32.0) ** 2
    r_mid = (mid / 32.0) ** 2
    if phases >= 3:
        red = f * (1 - r_lo) + f * (1 - r_mid)
    else:
        red = f * (1 - r_lo)
    t = steps * ms / 1000.0 * (1 - 0.78 * red)
    # accuracy: log-epoch gains with saturation, width losses
    acc = 0.7595
    if ep < 10:
        acc += 0.075 * math.log(ep / 10.0)
    else:
        acc += 0.03 * math.log(ep / 10.0)
    acc -= 0.015 * (math.log(128.0 / c[0]) + math.log(384.0 / c[1])
                    + 1.06 * math.log(576.0 / c[2]))
    if "translate4" in aug:
        acc -= 0.003
    if "cutout" in aug:
        acc -= 0.009
    if "alternating" not in aug and "random_flip" in aug:
        acc -= 0.004
    if ls > 0.1:
        acc -= 0.035 * (ls - 0.1)
    else:
        acc -= 0.025 * (0.1 - ls)
    if bs < 1024:
        acc -= 0.0005
    acc -= 0.012 * f
    if phases >= 3:
        acc -= 0.001
    return {"time": t, "accuracy": acc}
