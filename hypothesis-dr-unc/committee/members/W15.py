import math

def predict(config):
    g = config.get
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        w = [float(x) for x in ch]
    except Exception:
        w = [128.0, 384.0, 576.0]
    while len(w) < 3:
        w.append(w[-1])
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    steps = float(steps)
    # per-step cost (ms) at bs 1024: overhead + conv compute
    x = 8.29 + 4.83e-5 * (w[1] ** 2 / 1000.0) * 1000.0 / 1000.0 * 1000.0 / 1000.0 * 1.0
    x = 8.29 + 4.83e-5 * w[1] ** 2 + 8.7e-6 * w[2] ** 2
    fixed = 2.18
    step_ms = fixed + (bs / 1024.0) * (x - fixed)
    rm = str(g("resize_mode", "") or "")
    if rm:
        step_ms *= 0.83
    t = steps * step_ms / 1000.0

    aug = str(g("augmentation", "alternating_flip+translate2"))
    acc = 0.7585
    acc += 0.04 * math.log(max(ep, 1.0) / 10.0)
    acc -= 0.0148 * math.log(384.0 / w[1])
    acc -= 0.022 * math.log(576.0 / w[2])
    if "translate4" in aug:
        acc -= 0.0036
    if "cutout" in aug:
        acc -= 0.010
    if "alternating" not in aug:
        acc -= 0.004
    ls = float(g("label_smoothing", 0.1) or 0.0)
    if ls < 0.05:
        acc -= 0.003
    elif ls > 0.15:
        acc -= 0.0025
    if rm:
        acc -= 0.0055
    return {"time": t, "accuracy": acc}
