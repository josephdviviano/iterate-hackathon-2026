"""Apply I017: first conv group depth 3 -> 2 (per-group depths)."""
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''    "depth": 3,  # convs per group; depth 3 adds a residual around conv2/conv3''',
    '''    "depths": (2, 3, 3),  # convs per group; depth 3 adds a residual around conv2/conv3''')
rep('''    def __init__(self, num_classes, widths, depth, bn_momentum, scale):''',
    '''    def __init__(self, num_classes, widths, depths, bn_momentum, scale):''')
rep('''            ConvGroup(whiten_width, widths[0], depth, bn_momentum),
            ConvGroup(widths[0], widths[1], depth, bn_momentum),
            ConvGroup(widths[1], widths[2], depth, bn_momentum),''',
    '''            ConvGroup(whiten_width, widths[0], depths[0], bn_momentum),
            ConvGroup(widths[0], widths[1], depths[1], bn_momentum),
            ConvGroup(widths[1], widths[2], depths[2], bn_momentum),''')
rep('''hyp["widths"], hyp["depth"],''', '''hyp["widths"], hyp["depths"],''')
open(p, "w").write(s)
