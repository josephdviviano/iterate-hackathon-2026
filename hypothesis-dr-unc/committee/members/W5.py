import math

def predict(config: dict) -> dict:
    bs = config.get("batch_size", 1024) or 1024
    steps = config.get("train_steps", None)
    ep = config.get("epochs", 10)
    sub = config.get("train_subset_size", 50000) or 50000
    if not steps:
        steps = ep * sub / bs
    steps = max(steps, 1)
    aug = str(config.get("augmentation", ""))
    ls = config.get("label_smoothing", 0.1) or 0.0
    # time: fixed prepare/startup + per-step cost (scales with batch size)
    t = 0.6 + 0.0172 * steps * (bs / 1024.0)
    if "cutout" in aug:
        t += 0.02
    # accuracy: log-steps scaling around 488 steps, 1024 batch
    acc = 0.755 + 0.08 * math.log(max(steps * bs / 1024.0, 1) / 488.0)
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    if "translate" not in aug:
        acc -= 0.01
    acc -= 0.03 * abs(ls - 0.1)
    acc = min(acc, 0.9)
    return {"time": t, "accuracy": acc}
