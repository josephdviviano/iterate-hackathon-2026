import math

def predict(config):
    g = config.get
    def num(k, d):
        try:
            v = float(g(k, d))
            return v
        except Exception:
            return d
    ch = str(g('conv_channels', '128,384,576')).split(',')
    try:
        c2 = float(ch[1]); c3 = float(ch[2])
    except Exception:
        c2, c3 = 384.0, 576.0
    bs = num('batch_size', 1024)
    ep = num('epochs', 10)
    steps = num('train_steps', 0)
    if steps <= 0:
        steps = ep * 50000.0 / bs
    aug = str(g('augmentation', ''))
    rm = str(g('resize_mode', '') or '')
    # per-step cost (ms) at bs 1024
    f = 1.5 + 0.03125 * c2 + 0.0083 * c3
    f *= 0.15 + 0.85 * bs / 1024.0
    if 'cutout' in aug:
        f += 0.4
    frac = 0.0
    if rm:
        r = num('low_res_input_resolution', 24)
        frac = num('low_res_step_fraction', 0.5)
        f *= 1 - 0.73 * (1 - (r / 32.0) ** 2) * frac
    t = steps * f / 1000.0
    # accuracy
    a = 0.755
    a -= 0.0048 * (384 - c2) / 128 + 0.0104 * (576 - c3) / 192
    e = steps * bs / 50000.0
    if e >= 10:
        a += 0.0035 * (e - 10)
    else:
        a -= 0.0087 * (10 - e)
    if 'translate2' in aug:
        a += 0.003
    if 'cutout' in aug:
        a -= 0.006
    if 'random_flip' in aug:
        a -= 0.004
    ls = num('label_smoothing', 0.1)
    if ls > 0.1:
        a -= 0.06 * (ls - 0.1)
    else:
        a -= 0.03 * (0.1 - ls)
    if rm:
        a -= 0.01 * frac
    return {'time': t, 'accuracy': a}
