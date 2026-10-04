import math

def predict(config):
    g = config.get
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    steps = float(steps)
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        c2 = float(ch[1]); c3 = float(ch[2])
    except Exception:
        c2, c3 = 384.0, 576.0
    cost = 0.160 + 0.00304 * c2 + 0.000802 * c3
    aug = str(g("augmentation", ""))
    rs = str(g("resize_mode", "") or "")
    t = steps * bs * cost * 1e-5
    t *= 1.0 + 0.15 * (1024.0 / bs - 1.0)
    if "cutout" in aug:
        t *= 1.02
    if rs:
        t *= 0.82 if "antialias" in rs else 0.84
    ls = float(g("label_smoothing", 0.1) or 0.0)
    acc = 0.7545
    acc += 0.05 * math.log(max(ep, 1.0) / 10.0)
    acc -= 0.025 * max(0.0, 1.789 - cost)
    acc -= 0.03 * abs(ls - 0.1)
    if "cutout" in aug:
        acc -= 0.0095
    if "translate2" in aug:
        acc += 0.0035
    if "random_flip" in aug:
        acc -= 0.004
    if rs:
        acc -= 0.0055
    if bs < 1024:
        acc -= 0.0004
    return {"time": t, "accuracy": acc}
