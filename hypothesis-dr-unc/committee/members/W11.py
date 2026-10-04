import math

def predict(config):
    p = config.get("num_params_approx", 2500000) or 2500000
    ep = config.get("epochs", 10) or 10
    bs = config.get("batch_size", 1024) or 1024
    steps = config.get("train_steps", None)
    if not steps:
        steps = ep * 50000.0 / bs
    else:
        ep = steps * bs / 50000.0
    aug = str(config.get("augmentation", "alternating_flip+translate4"))
    ls = config.get("label_smoothing", 0.1)
    # time: per-step cost affine in params
    t = steps * (0.00822 + 4.07e-9 * p) * (1024.0 / bs) ** 0 + 0.05
    # accuracy
    acc = 0.7707
    acc -= 0.86 * math.exp(-ep / 2.5)
    acc -= 0.0125 * math.log(2500000.0 / p, 2)
    if "cutout" in aug:
        acc -= 0.0115
    if "random_flip" in aug:
        acc -= 0.004
    if "alternating" not in aug and "random_flip" not in aug:
        acc -= 0.006
    acc -= 0.02 * abs(ls - 0.1)
    return {"time": t, "accuracy": acc}
