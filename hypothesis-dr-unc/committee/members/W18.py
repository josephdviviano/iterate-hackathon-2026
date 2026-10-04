import math

def predict(config):
    g = config.get
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 10) or 10)
    steps = g("train_steps", None)
    steps = float(steps) if steps else ep * 50000.0 / bs
    if not g("epochs", None):
        ep = steps * bs / 50000.0
    ch = str(g("conv_channels", "128,384,576")).split(",")
    try:
        c = [float(x) for x in ch]
        c2, c3 = c[1], c[2]
    except Exception:
        c2, c3 = 384.0, 576.0
    cost = 0.114 + 0.00217 * c2 + 0.000573 * c3
    aug = str(g("augmentation", "alternating_flip+translate2"))
    rs = str(g("resize_mode", "") or "")
    per = 0.00215 + 1.19e-5 * bs
    t = steps * per * cost
    if rs:
        t *= 0.835
    if "cutout" in aug:
        t *= 1.025
    # accuracy
    a = 0.7585
    a += 0.005 * (ep - 10.0)
    a -= 0.0068 * (384.0 - c2) / 128.0 + 0.0080 * (576.0 - c3) / 192.0
    if "translate4" in aug:
        a -= 0.0035
    if "cutout" in aug:
        a -= 0.0095
    if "random_flip" in aug:
        a -= 0.0042
    ls = float(g("label_smoothing", 0.1) or 0.0)
    a -= 0.3 * (ls - 0.1) ** 2
    if rs:
        a -= 0.006
    return {"time": t, "accuracy": a}
