import math

def predict(config):
    g = config.get
    cc = g("conv_channels", "128,384,576")
    try:
        c = [float(x) for x in str(cc).split(",")]
    except Exception:
        c = [128.0, 384.0, 576.0]
    while len(c) < 3:
        c.append(c[-1])
    c1, c2, c3 = c[:3]
    f = c1*c1 + (c1*c2 + c2*c2)*0.25 + (c2*c3 + c3*c3)/16.0
    ep = float(g("epochs", 10) or 10)
    b = float(g("batch_size", 1024) or 1024)
    steps = g("train_steps", None)
    if not steps:
        steps = ep*50000.0/b
    steps = float(steps)
    aug = str(g("augmentation", "alternating_flip+translate2"))
    ls = float(g("label_smoothing", 0.1) or 0.0)
    frac = float(g("low_res_step_fraction", 0) or 0.0)
    nph = g("num_resolution_phases", None)
    try:
        nph = int(nph)
    except Exception:
        nph = 2 if frac > 0 else 1
    bs = (b/1024.0)**0.9
    comp = f*1.27e-4*bs
    over = 5.6*bs
    cut = 0.6*bs if "cutout" in aug else 0.0
    tot = comp + over + cut
    t = 0.1 + steps*tot/1000.0
    if frac > 0:
        sv = frac*0.4375
        if nph >= 3:
            sv += frac*0.234
        t -= 1.25*steps*comp/1000.0*sv
    acc = 0.7585
    acc += 0.06*math.log(ep/10.0)
    acc -= 0.032*max(0.0, 1.0 - f/100096.0)
    if "cutout" in aug:
        acc -= 0.012
    if "random_flip" in aug:
        acc -= 0.004
    if "translate4" in aug:
        acc += -0.0035
    elif "translate2" not in aug:
        acc += -0.002
    if ls < 0.05:
        acc -= 0.003
    elif ls > 0.15:
        acc -= 0.002
    acc -= 0.014*frac
    return {"time": t, "accuracy": acc}
