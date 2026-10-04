import math

def predict(config):
    g = config.get
    def num(k, d):
        v = g(k, d)
        try:
            return float(v)
        except Exception:
            return d
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        w1, w2, w3 = [float(x) for x in ch[:3]]
    except Exception:
        w1, w2, w3 = 128., 384., 576.
    bs = num("batch_size", 1024)
    ep = num("epochs", 10)
    steps = ep * 50000.0 / bs
    f = (w1 * w1 * 1024 + (w1 * w2 + w2 * w2) * 256 + (w2 * w3 + w3 * w3) * 64) / 1e6
    step_t = (0.00555 + 1.25e-4 * f) * (0.2 + 0.8 * bs / 1024.0)
    frac = num("low_res_step_fraction", 0) if g("low_res_step_fraction", "") not in ("", None) else 0.0
    ph = int(num("num_resolution_phases", 2)) if g("num_resolution_phases", "") not in ("", None) else 2
    def r(res):
        return 0.2 + 0.8 * (res / 32.0) ** 2
    lo = r(num("low_res_input_resolution", 24))
    mid = r(num("mid_res_input_resolution", 28))
    mult = 1.0 - frac * (1 - lo)
    if ph >= 3:
        mult -= frac * (1 - mid)
    time = steps * step_t * mult

    aug = str(g("augmentation", "alternating_flip+translate2"))
    acc = 0.7585 - 0.026 * math.log(102.5 / f)
    acc += 0.006 * (ep - 10) if ep >= 10 else 0.0065 * (ep - 10)
    if "translate4" in aug:
        acc -= 0.003
    if "cutout" in aug:
        acc -= 0.006
    if "random_flip" in aug:
        acc -= 0.004
    ls = num("label_smoothing", 0.1)
    if ls > 0.15:
        acc -= 0.005 * (ls - 0.1) / 0.1
    elif ls < 0.05:
        acc -= 0.003
    acc -= 0.020 * frac * (1.3 if ph >= 3 else 1.0)
    if str(g("pooling", "global_max")).startswith("adaptive"):
        acc -= 0.002
    return {"time": time, "accuracy": acc}
