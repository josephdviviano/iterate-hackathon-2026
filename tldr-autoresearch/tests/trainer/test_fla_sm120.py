"""FLA 0.5.2 gated-DeltaNet kernel sanity on sm_120 (RTX PRO 6000 Blackwell): forward + backward vs the fp32
naive recurrent reference, then NaN/timing checks at training lengths. Shapes match Qwen3.8-27B DeltaNet layers
(48 value heads, head dim 128, q/k L2-normalised in kernel, fp32 log-decay g, bf16 beta).

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<trainer GPU UUID> \
        ~/envs/train/bin/python tests/trainer/test_fla_sm120.py
"""
import sys
import time

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

DEV = "cuda"
H, K, V = 48, 128, 128


def make(T, seed=0):
    g_ = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(1, T, H, K, device=DEV, generator=g_).bfloat16().requires_grad_(True)
    k = torch.randn(1, T, H, K, device=DEV, generator=g_).bfloat16().requires_grad_(True)
    v = torch.randn(1, T, H, V, device=DEV, generator=g_).bfloat16().requires_grad_(True)
    a = torch.randn(1, T, H, device=DEV, generator=g_)
    decay = torch.exp(torch.randn(H, device=DEV, generator=g_) * 0.5)
    g = (-decay * F.softplus(a)).detach().requires_grad_(True)           # log-decay, fp32 (as in the model)
    beta = torch.rand(1, T, H, device=DEV, generator=g_).bfloat16().sigmoid().detach().requires_grad_(True)
    return q, k, v, g, beta


def main():
    ok = True
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "torch", torch.__version__)
    T = 1024
    q, k, v, g, beta = make(T)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    do = torch.randn_like(o)
    (o * do).sum().backward()
    grads = [t.grad.clone() for t in (q, k, v, g, beta)]
    for t in (q, k, v, g, beta):
        t.grad = None
    ref, _ = naive_recurrent_gated_delta_rule(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1),
                                              v.float(), beta.float(), g.float(), scale=K ** -0.5)
    (ref * do.float()).sum().backward()
    fwd_err = ((o.float() - ref).norm() / ref.norm()).item()
    ok &= fwd_err < 1e-2
    print(f"T={T} fwd rel err vs fp32 naive: {fwd_err:.3e}")
    for n, a_, t in zip("q k v g beta".split(), grads, (q, k, v, g, beta)):
        e = ((a_.float() - t.grad.float()).norm() / t.grad.float().norm()).item()
        nan = torch.isnan(a_).any().item()
        ok &= (e < 1e-2) and not nan
        print(f"  grad {n:4s}: rel err {e:.3e} nan={nan}")
    for T in (8192, 32768, 65536):
        q, k, v, g, beta = make(T, seed=T)
        torch.cuda.synchronize()
        t0 = time.time()
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        tf = time.time() - t0
        t0 = time.time()
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        tb = time.time() - t0
        bad = any(torch.isnan(x.grad).any().item() or torch.isinf(x.grad).any().item() for x in (q, k, v, g, beta))
        ok &= not bad and not torch.isnan(o).any().item()
        print(f"T={T}: fwd {tf * 1e3:.1f} ms, bwd {tb * 1e3:.1f} ms (first call incl. autotune), nan/inf={bad}")
        del q, k, v, g, beta, o
    print("FLA sm_120 sanity:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
