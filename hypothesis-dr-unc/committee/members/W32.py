import math

def _f(c, k, d):
    v = c.get(k, d)
    try:
        if v is None or v == "":
            return d
        return float(v)
    except Exception:
        return d

def predict(config):
    c = config
    bs = _f(c, "batch_size", 1024.0)
    ep = _f(c, "epochs", 10.0)
    steps = _f(c, "train_steps", 0.0)
    if steps <= 0:
        steps = ep * 50000.0 / bs
    ep_eff = steps * bs / 50000.0
    try:
        w = [float(x) for x in str(c.get("conv_channels", "128,256,576")).split(",")]
    except Exception:
        w = [128.0, 256.0, 576.0]
    while len(w) < 3:
        w.append(w[-1])
    c1, c2, c3 = w[0], w[1], w[2]
    lf = min(max(_f(c, "low_res_step_fraction", 0.0), 0.0), 1.0)
    phases = _f(c, "num_resolution_phases", 2.0)
    opt = str(c.get("optimizer", "sgd_lookahead_ema"))
    aug = str(c.get("augmentation", "alternating_flip+translate2"))
    ls = _f(c, "label_smoothing", 0.1)

    # per-step cost (ms at batch 1024): affine in stage widths
    per = 0.042 * c1 + 0.031 * c2 + 0.0083 * c3 - 5.1
    per = max(per, 3.0) * (bs / 1024.0) ** 0.8
    # low-res phases cut compute by pixel ratio (24px: 0.4375, 28px: 0.234 saved)
    save = 0.4375 * lf
    if phases >= 3:
        save += 0.234 * lf
    per *= 1.0 - 0.78 * save
    over = 0.0
    if "muon" in opt:
        over += 2.26 if opt.startswith("sgd") else 1.45
    if "cutout" in aug:
        over += 0.2
    t = 0.7 + steps * (per + over) / 1000.0

    # accuracy: relative to 128,256,576 / 11 epochs / translate2 / ls 0.1
    a = 0.7567
    d = ep_eff - 11.0
    if d >= 0:
        a += 0.003 * d
    elif ep_eff >= 10.0:
        a += 0.0052 * d
    else:
        a += -0.0052 + 0.0085 * (ep_eff - 10.0)
    a += 0.00016 * (c1 - 128.0) + 0.000053 * (c2 - 256.0)
    a += 0.000043 * (min(c3, 576.0) - 576.0) + 0.00002 * max(c3 - 576.0, 0.0)
    if "muon" in opt:
        a += 0.008
    if "cutout" in aug:
        a -= 0.0095
    if "translate4" in aug:
        a -= 0.0035
    if "random_flip" in aug:
        a -= 0.004
    a -= 0.03 * abs(ls - 0.1)
    if lf > 0:
        a -= 0.0035 + 0.004 * lf
    a = min(a, 0.80)
    return {"time": t, "accuracy": a}
