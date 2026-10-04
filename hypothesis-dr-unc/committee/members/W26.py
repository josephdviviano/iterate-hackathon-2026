import math

def _ch(c):
    if isinstance(c, str):
        c = [float(x) for x in c.replace(' ', '').split(',') if x]
    c = list(c) if c else [128, 384, 576]
    while len(c) < 3:
        c.append(c[-1])
    return [float(x) for x in c[:3]]

def predict(config):
    g = config.get
    c1, c2, c3 = _ch(g('conv_channels', '128,384,576'))
    w = (16 * c1 * c1 + 4 * c2 * c2 + 0.5 * c3 * c3) / 1e6
    ep = float(g('epochs', 10) or 10)
    bs = float(g('batch_size', 1024) or 1024)
    frac = float(g('low_res_step_fraction', 0) or 0)
    aug = str(g('augmentation', 'alternating_flip+translate4'))
    ls = float(g('label_smoothing', 0.1) or 0)
    opt = str(g('optimizer', 'sgd_lookahead_ema'))
    cost = 0.245 + 0.645 * w
    t = ep * cost * (1 - 0.36 * frac) * (1024.0 / bs) ** 0.1
    if 'muon' in opt:
        t += 0.8
    acc = 0.755 + 0.07 * math.log(ep / 10.0) + 0.03 * (w - 1.018)
    acc -= 0.011 * frac
    if 'translate2' in aug:
        acc += 0.0035
    elif 'translate4' not in aug:
        acc -= 0.002
    if 'cutout' in aug:
        acc -= 0.006
    if 'random_flip' in aug:
        acc -= 0.004
    acc -= 0.05 * abs(ls - 0.1) if ls < 0.1 else 0.0
    if ls > 0.1:
        acc -= 0.055 * (ls - 0.1)
    if bs < 1024:
        acc -= 0.0004
    elif bs > 1024:
        acc -= 0.0005
    if 'muon' in opt:
        acc += 0.0078
    return {'time': t, 'accuracy': acc}
