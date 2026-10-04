import math

def predict(config):
    g = config.get
    def num(k, d):
        try:
            v = float(g(k, d))
            return v
        except Exception:
            return d
    bs = num('batch_size', 1024)
    ep = num('epochs', 10)
    steps = num('train_steps', 0)
    if steps <= 0:
        steps = ep * 50000.0 / bs
    ch = str(g('conv_channels', '128,384,576')).split(',')
    try:
        c = [float(x) for x in ch]
    except Exception:
        c = [128.0, 384.0, 576.0]
    while len(c) < 3:
        c.append(c[-1])
    c2, c3 = c[1], c[2]
    ls = num('label_smoothing', 0.1)
    aug = str(g('augmentation', ''))
    rm = str(g('resize_mode', ''))
    # time
    per = 1.5 + 0.03125 * c2 + 0.00833 * c3
    per *= (0.12 + 0.88 * bs / 1024.0)
    if rm not in ('', 'none', 'None'):
        frac = num('low_res_step_fraction', 0.5)
        res = num('low_res_input_resolution', 24)
        r = 0.32 + 0.68 * (res / 24.0) ** 2 * 0.0 + 0.0
        r = 0.68 * (res / 24.0) ** 2
        per *= 1 - frac * (1 - r)
    t = 0.15 + steps * per / 1000.0
    # accuracy
    a = 0.7545
    if ep < 10:
        a += 0.0087 * (ep - 10)
    else:
        a += 0.0045 * (ep - 10)
    if 'translate2' in aug:
        a += 0.004
    if 'cutout' in aug:
        a -= 0.008
    if 'random_flip' in aug:
        a -= 0.004
    a -= 0.03 * abs(ls - 0.1) if ls > 0.1 else 0.03 * (0.1 - ls)
    a -= 0.0145 * 0.585 * max(0.0, 384 - c2) / 128.0
    a -= 0.0085 * max(0.0, 576 - c3) / 192.0
    if rm not in ('', 'none', 'None'):
        a -= 0.0055 * num('low_res_step_fraction', 0.5) / 0.5
    return {'time': t, 'accuracy': a}
