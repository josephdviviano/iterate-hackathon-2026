import math

def predict(config):
    c = config
    ep = float(c.get("epochs", 10))
    bs = float(c.get("batch_size", 1024))
    steps = float(c.get("train_steps", ep * 50000 / bs))
    p = float(c.get("num_params_approx", 2500000)) / 1e6
    aug = str(c.get("augmentation", "alternating_flip+translate4"))
    ls = float(c.get("label_smoothing", 0.1))
    per_step = 0.00846 + 0.00394 * p
    t = steps * per_step
    if "cutout" in aug:
        t *= 1.03
    acc = 0.7545
    acc += 0.05 * math.log(max(steps, 1) / 488.0)
    acc += 0.0177 * math.log(max(p, 0.05) / 2.5)
    if "cutout" in aug:
        acc -= 0.011
    if "random_flip" in aug:
        acc -= 0.004
    if "flip" not in aug:
        acc -= 0.01
    acc -= 0.02 * abs(ls - 0.1)
    return {"time": t, "accuracy": min(acc, 0.99)}
