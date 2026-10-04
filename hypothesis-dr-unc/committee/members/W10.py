import math

def _chs(ch):
    if isinstance(ch, (int, float)):
        return [float(ch)] * 3
    if isinstance(ch, str):
        cs = []
        for x in ch.split(","):
            try:
                cs.append(float(x))
            except ValueError:
                pass
    else:
        try:
            cs = [float(x) for x in ch]
        except Exception:
            cs = []
    return cs or [128.0, 384.0, 576.0]

def predict(config: dict) -> dict:
    cs = _chs(config.get("conv_channels", "128,384,576"))
    s2 = sum(c * c for c in cs)
    bs = float(config.get("batch_size", 1024) or 1024)
    steps = config.get("train_steps")
    if not steps:
        ep = float(config.get("epochs", 10) or 10)
        steps = ep * 50000.0 / bs
    steps = float(steps)
    per_step = 0.0081 + 2.03e-8 * s2
    if bs < 1024:
        per_step *= bs / 1024.0
    t = steps * per_step
    aug = str(config.get("augmentation", "alternating_flip+translate4"))
    try:
        ls = float(config.get("label_smoothing", 0.1))
    except Exception:
        ls = 0.1
    acc = 0.755
    acc += 0.0195 * math.log(s2 / 495616.0)
    acc += 0.05 * math.log(steps / 488.0)
    if "cutout" in aug:
        acc -= 0.008
    if "random_flip" in aug:
        acc -= 0.004
    if "alternating" not in aug and "random_flip" not in aug:
        acc -= 0.006
    acc -= 0.04 * abs(ls - 0.1)
    return {"time": t, "accuracy": acc}
