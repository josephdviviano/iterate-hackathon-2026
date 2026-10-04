import math

def predict(config):
    g = config.get
    def num(k, d):
        v = g(k)
        try:
            return float(v) if v not in (None, "") else d
        except Exception:
            return d
    ch = str(g("conv_channels", "128,256,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128., 256., 576.
    bs = num("batch_size", 1024)
    ep = num("epochs", 10)
    steps = num("train_steps", ep * 50000 / bs)
    aug = str(g("augmentation", ""))
    ls = num("label_smoothing", 0.1)
    M = (1024 * 9 * c1 * c1 + 2304 * (c1 * c2 + c2 * c2) + 576 * (c2 * c3 + c3 * c3)) / 1e6
    # low-res curriculum
    rm = g("resize_mode")
    phases = int(num("num_resolution_phases", 2)) if rm else 1
    sav = 0.0
    if rm:
        if phases >= 3:
            f = num("low_res_step_fraction", 1 / 3.)
            lo = num("low_res_input_resolution", 24)
            mid = num("mid_res_input_resolution", 28)
            sav = f * (1 - (lo / 32.) ** 2) + f * (1 - (mid / 32.) ** 2)
        else:
            f = num("low_res_step_fraction", 0.5)
            lo = num("low_res_input_resolution", 24)
            sav = f * (1 - (lo / 32.) ** 2)
    step_cost = 0.0019 + 1.45e-5 * M
    step_cost *= bs / 1024.0
    step_cost *= (1 - 0.75 * sav)
    t = 0.4 + steps * (0.0022 + step_cost)
    if "cutout" in aug:
        t += 0.3
    # accuracy
    acc = 0.7585
    if ep < 10:
        acc -= 0.009 * (10 - ep)
    else:
        acc += 0.0028 * (ep - 10)
    acc -= 0.000035 * max(0, 384 - c2)
    acc -= 0.000052 * max(0, 576 - c3)
    acc -= 0.00014 * max(0, 128 - c1)
    if "translate4" in aug:
        acc -= 0.0035
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    if ls > 0.15:
        acc -= 0.004
    elif ls < 0.05:
        acc -= 0.003
    acc -= 0.0252 * sav
    return {"time": t, "accuracy": acc}
