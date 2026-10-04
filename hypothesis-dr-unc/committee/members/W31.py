import math

def predict(config):
    g = config.get
    ch = str(g("conv_channels", "128,256,576")).split(",")
    try:
        c1, c2, c3 = [float(x) for x in ch[:3]]
    except Exception:
        c1, c2, c3 = 128.0, 256.0, 576.0
    bs = float(g("batch_size", 1024) or 1024)
    ep = float(g("epochs", 11) or 11)
    steps = g("train_steps", None)
    if not steps:
        steps = ep * 50000 / bs
    steps = float(steps)
    lf = float(g("low_res_step_fraction", 0) or 0)
    try:
        nph = int(g("num_resolution_phases", 2) or 2)
    except Exception:
        nph = 2
    opt = str(g("optimizer", "sgd_lookahead_ema"))
    # per-sample compute scaled by width
    per = -0.0040 + 4.4e-5 * c1 + 3.1e-5 * c2 + 8.2e-6 * c3
    per = max(per, 0.004)
    # low-res phases cut pixel compute: 24px costs 0.5625, 28px 0.766
    sav = lf * (1 - 0.5625)
    if nph >= 3:
        sav += lf * (1 - 0.766)
    f = 1 - 0.82 * sav
    # per-step fixed overhead (~12% of a 1024-batch step)
    stepf = 0.88 * bs / 1024.0 + 0.12
    t = steps * stepf * per * f
    if "muon" in opt:
        t += 0.8 if opt.startswith("sgd") else 0.5
    # accuracy
    a = 0.7567
    d = ep - 11
    a += (0.004 if d < 0 else 0.002) * d
    a += 0.00005 * (c2 - 256) + 0.00003 * (c3 - 576) + 0.00016 * (c1 - 128)
    aug = str(g("augmentation", "alternating_flip+translate2"))
    if "cutout" in aug:
        a -= 0.008
    if "translate4" in aug:
        a -= 0.003
    if "random_flip" in aug:
        a -= 0.004
    ls = float(g("label_smoothing", 0.1) or 0)
    if ls < 0.05:
        a -= 0.003
    elif ls > 0.15:
        a -= 0.004
    a -= 0.007 * min(lf / 0.33, 1.0) if lf > 0 else 0
    if "muon" in opt:
        a += 0.007
    if bs < 1024:
        a -= 0.0004
    elif bs > 1024:
        a -= 0.0005
    return {"time": t, "accuracy": a}
