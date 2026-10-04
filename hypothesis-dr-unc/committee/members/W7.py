import math

def predict(config):
    g = config.get
    try:
        c = [float(x) for x in str(g("conv_channels", "128,384,576")).split(",")]
    except Exception:
        c = [128.0, 384.0, 576.0]
    bs = float(g("batch_size", 1024) or 1024)
    steps = g("train_steps", None)
    if not steps:
        steps = float(g("epochs", 10)) * float(g("train_subset_size", 50000)) / bs
    steps = float(steps)
    F = (sum(x * x for x in c[1:]) if len(c) > 1 else c[0] ** 2) / 1000.0
    res = float(g("input_resolution", 32)) / 32.0
    lowr = float(g("low_res_input_resolution", 24)) / 32.0
    lf = float(g("low_res_step_fraction", 0.5))
    rs = res * res * ((1 - lf) + lf * lowr * lowr) / 0.78
    # per-step: fixed overhead (launch, aug, optimizer) + conv compute scaling with width^2
    per = 0.0081 * (bs / 1024.0) ** 0.5 + 2.17e-5 * F * (bs / 1024.0) * rs
    t = steps * per
    if not g("torch_compile", True):
        t *= 1.3
    acc = 0.755
    acc += 0.05 * math.log(max(steps, 1) / 488.0)
    acc += 0.018 * math.log(max(F, 1.0) / 479.0)
    aug = str(g("augmentation", "alternating_flip+translate4"))
    if "cutout" in aug:
        acc -= 0.0125
    if "random_flip" in aug:
        acc -= 0.004
    ls = float(g("label_smoothing", 0.1))
    if ls > 0.1:
        acc -= 0.01 * (ls - 0.1)
    if not g("whitening_init", True):
        acc -= 0.01
    if not g("ema", True):
        acc -= 0.003
    return {"time": t, "accuracy": min(acc, 0.95)}
