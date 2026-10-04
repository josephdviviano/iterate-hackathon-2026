import math

def predict(config):
    g = config.get
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        c = [float(x) for x in ch][:3]
        while len(c) < 3:
            c.append(c[-1])
    except Exception:
        c = [128.0, 384.0, 576.0]
    f = sum(x * x for x in c)
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    steps = float(steps)
    res = float(g("input_resolution", 32) or 32)
    r2 = (res / 32.0) ** 2
    per = (0.00758 + 2.08e-8 * f) * r2 * (bs / 1024.0)
    aug = str(g("augmentation", "alternating_flip+translate4"))
    if "cutout" in aug:
        per += 0.0006
    t = 0.2 + steps * per
    if g("torch_compile", True) is False:
        t *= 1.3
    ls = float(g("label_smoothing", 0.1) or 0.0)
    acc = 0.755
    acc += 0.045 * math.log(max(ep, 1.0) / 10.0)
    acc += 0.0195 * math.log(f / 495616.0)
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    if "translate" not in aug:
        acc -= 0.03
    acc -= 0.04 * max(0.0, ls - 0.1)
    acc -= 0.1 * abs(res - 32) / 32.0 if res < 32 else 0.0
    if not g("ema", True):
        acc -= 0.003
    if not g("whitening_init", True):
        acc -= 0.005
    return {"time": t, "accuracy": min(acc, 0.95)}
