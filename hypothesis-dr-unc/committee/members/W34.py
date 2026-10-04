import math

def predict(config):
    g = config.get
    try:
        w1, w2, w3 = [float(x) for x in str(g("conv_channels", "128,256,576")).split(",")[:3]]
    except Exception:
        w1, w2, w3 = 128.0, 256.0, 576.0
    ep = float(g("epochs", 11) or 11)
    bs = float(g("batch_size", 1024) or 1024)
    f = float(g("low_res_step_fraction", 0) or 0)
    r = float(g("low_res_input_resolution", 24) or 24)
    mr = float(g("mid_res_input_resolution", 28) or 28)
    ph = float(g("num_resolution_phases", 2) or 2)
    opt = str(g("optimizer", "sgd_lookahead_ema"))
    aug = str(g("augmentation", "alternating_flip+translate2"))
    pool = str(g("pooling", "global_max"))
    ls = float(g("label_smoothing", 0.1) or 0)
    phases = [(f, r)]
    if ph >= 3 and f > 0:
        phases.append((f, mr))
    # time: per-epoch compute grows with stage widths; low-res phases cut compute
    p = 0.70 + 0.0021 * (w1 - 128) + 0.0014 * (w2 - 256) + 0.0004 * (w3 - 576)
    sav = sum(fi * (1 - (ri / 32.0) ** 2) for fi, ri in phases)
    lowres = max(0.3, 1 - 0.82 * sav)
    bsf = 1 - 0.06 * math.log(bs / 1024.0, 2)
    of = 1.0
    if "muon_conv" in opt:
        of = 1.07
    elif opt.startswith("muon"):
        of = 1.08
    elif "muon" in opt:
        of = 1.13
    t = ep * p * lowres * bsf * of
    # accuracy
    a = 0.7567
    d = math.log(max(ep, 1) / 11.0)
    a += (0.05 if d < 0 else 0.02) * d
    a += 0.00005 * (w2 - 256) + 0.00015 * (w1 - 128) + 0.00004 * (w3 - 576)
    if "cutout" in aug:
        a -= 0.009
    if "translate4" in aug:
        a -= 0.0035
    if "random_flip" in aug:
        a -= 0.004
    if ls <= 0.01:
        a -= 0.003
    elif ls > 0.15:
        a -= 0.003
    x = sum(fi * (1 - ri / 32.0) for fi, ri in phases)
    if x > 0:
        a -= 0.007 * (x / 0.125) ** 2.0
    if "adaptive" in pool:
        a -= 0.001
    a -= 0.0015 * math.log(bs / 1024.0, 2)
    if "muon_conv" in opt:
        a += 0.0115
    elif "muon" in opt:
        a += 0.0082
    return {"time": t, "accuracy": a}
