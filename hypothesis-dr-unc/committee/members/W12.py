import math

def predict(config: dict) -> dict:
    g = config.get
    bs = g("batch_size", 1024) or 1024
    steps = g("train_steps", None)
    epochs = g("epochs", 10)
    if steps is None:
        steps = epochs * 50000 / bs
    else:
        epochs = steps * bs / 50000.0
    params = float(g("num_params_approx", 2500000) or 2500000)
    aug = str(g("augmentation", "alternating_flip+translate4"))
    ls = g("label_smoothing", 0.1) or 0.0
    sub = g("train_subset_size", 50000) or 50000
    cut = "cutout" in aug
    per_step = (0.0083 + 4e-9 * params) * (bs / 1024.0)
    if cut:
        per_step += 0.0005
    if "random_flip" in aug:
        per_step += 0.00015
    t = steps * per_step
    if g("torch_compile", True) is False:
        t *= 1.3
    acc = 0.755
    acc += 0.018 * math.log(params / 2.5e6)
    coeff = 0.06 if params > 2e6 else 0.03
    acc += coeff * math.log(max(epochs, 0.5) / 10.0)
    if cut:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    acc -= 0.04 * max(0.0, ls - 0.1)
    acc -= 0.05 * max(0.0, 1 - sub / 50000.0)
    return {"time": t, "accuracy": acc}
