import math

def _chan(cc):
    if isinstance(cc, str):
        cc = [float(x) for x in cc.split(",") if x.strip()]
    elif isinstance(cc, (int, float)):
        cc = [float(cc)]
    else:
        try:
            cc = [float(x) for x in cc]
        except Exception:
            cc = []
    if not cc:
        cc = [128.0, 384.0, 576.0]
    if len(cc) == 1:
        cc = [128.0, cc[0], cc[0] * 1.5]
    while len(cc) < 3:
        cc.append(cc[-1])
    return cc[:3]

def _num(v, d):
    try:
        if v is None or v == "":
            return d
        return float(v)
    except Exception:
        return d

def predict(config):
    g = config.get
    c0, c1, c2 = _chan(g("conv_channels", "128,384,576"))
    bs = _num(g("batch_size", 1024), 1024.0) or 1024.0
    ep_cfg = _num(g("epochs", 10), 10.0)
    steps = _num(g("train_steps", None), 0.0)
    if steps <= 0:
        steps = ep_cfg * 50000.0 / bs
    # steps are authoritative: effective epochs
    ep = steps * bs / 50000.0
    f = _num(g("low_res_step_fraction", 0), 0.0)
    phases = _num(g("num_resolution_phases", 0), 0.0)
    aug = str(g("augmentation", "alternating_flip+translate2"))
    ls = _num(g("label_smoothing", 0.1), 0.1)
    opt = str(g("optimizer", "sgd_lookahead_ema"))
    muon = "muon" in opt

    wf = 1.0 - 0.22 * (384.0 - c1) / 128.0 + 0.0004 * (c2 - 576.0) \
        - 0.08 * (128.0 - c0) / 32.0
    wf = max(wf, 0.2)
    resf = 1.0 - 0.78 * f * (1.0 - (24.0 / 32.0) ** 2)
    if phases >= 3:
        resf *= 0.95
    cut = 1.03 if "cutout" in aug else 1.0
    bsf = (1024.0 / bs) ** 0.1
    t = 0.25 + steps * bs * 1.74e-5 * wf * resf * cut * bsf
    if muon:
        t += 0.8

    a = 0.7585
    if "translate4" in aug:
        a -= 0.0035
    if "random_flip" in aug:
        a -= 0.004
    if "cutout" in aug:
        a -= 0.008
    a += 0.07 * math.log(max(ep, 1.0) / 10.0)
    a += 0.019 * math.log((c1 * c2) / (384.0 * 576.0))
    a += 0.015 * math.log(c0 / 128.0)
    a -= 0.012 * f
    if ls < 0.05:
        a -= 0.003
    elif ls > 0.15:
        a -= 0.003
    if muon:
        a += 0.0075
    return {"time": t, "accuracy": a}
