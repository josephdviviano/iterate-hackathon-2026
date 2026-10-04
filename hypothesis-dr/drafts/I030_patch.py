"""Apply I030: frequency-wise (DCT-II 3x3 basis) Muon-C split instead of flattened [out, in*9] orthogonalization.

Each 3x3 kernel momentum is projected onto the orthonormal separable DCT-II basis -> 9 [out, in] matrices per
filter, orthogonalized together with one batched NS, then transformed back. Scaling: sqrt(max(1, out/in)) on the
[out, in] shape, and an extra 1/3 so each filter row's update norm matches flattened Muon (9 orthonormal frequency
slices each give unit-norm rows, i.e. sqrt(9) = 3x the flattened row norm).
"""
import math
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''@torch.compile(dynamic=False)
def _muon_update(params, grads, bufs, lr, momentum: float, ns_steps: int, shape_groups: tuple):
    with torch.no_grad():
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        updates = torch._foreach_add(grads, bufs, alpha=momentum)
        for idx in shape_groups:
            G = torch.stack([updates[i].reshape(len(updates[i]), -1) for i in idx])
            scale = max(1.0, G.size(1) / G.size(2)) ** 0.5
            U = _newtonschulz(G, ns_steps)
            for j, i in enumerate(idx):
                p = params[i]
                p.mul_(len(p) ** 0.5 / p.norm())
                p.sub_(U[j].view(p.shape).float() * (lr * scale))''', '''def dct3_basis(device):
    """Orthonormal 3-point DCT-II matrix D[k, n]."""
    n = torch.arange(3, dtype=torch.float64)
    D = torch.stack([torch.cos(math.pi * (n + 0.5) * k / 3) for k in range(3)])
    D[0] *= math.sqrt(1 / 3)
    D[1:] *= math.sqrt(2 / 3)
    return D.float().to(device)


@torch.compile(dynamic=False)
def _muon_update(params, grads, bufs, lr, D, momentum: float, ns_steps: int, shape_groups: tuple):
    with torch.no_grad():
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        updates = torch._foreach_add(grads, bufs, alpha=momentum)
        for idx in shape_groups:
            G = torch.stack([updates[i] for i in idx])  # [p, out, in, 3, 3]
            k, out, cin = len(idx), G.size(1), G.size(2)
            F = torch.einsum("an,bm,poinm->paboi", D, D, G).reshape(k * 9, out, cin)
            scale = max(1.0, out / cin) ** 0.5 / 3.0
            U = _newtonschulz(F, ns_steps).float().reshape(k, 3, 3, out, cin)
            U = torch.einsum("an,bm,paboi->poinm", D, D, U)
            for j, i in enumerate(idx):
                p = params[i]
                p.mul_(len(p) ** 0.5 / p.norm())
                p.sub_(U[j] * (lr * scale))''')
rep('''        self.lr_t = torch.tensor(float(lr), device=self.params[0].device)''', '''        self.lr_t = torch.tensor(float(lr), device=self.params[0].device)
        self.D = dct3_basis(self.params[0].device)''')
rep('''        _muon_update(self.params, grads, self.bufs, self.lr_t, self.momentum, self.ns_steps, self.shape_groups)''',
    '''        _muon_update(self.params, grads, self.bufs, self.lr_t, self.D, self.momentum, self.ns_steps, self.shape_groups)''')
if "import math" not in s:
    rep('''from types import SimpleNamespace''', '''import math
from types import SimpleNamespace''')
open(p, "w").write(s)
