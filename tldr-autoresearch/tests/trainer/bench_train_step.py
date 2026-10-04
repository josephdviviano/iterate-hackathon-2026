"""Memory / time benchmark of one full training micro-step on the real 64-layer policy (GPU 3 only):
forward with grad on the LoRA, token_logprobs over every position, a GRPO-like weighted loss, backward.
A fused AdamW step is taken once up front so optimizer state is resident during all measurements.

For each mode the sequence length is increased until OOM; the largest length that ran is reported together with
its peak allocated / reserved memory (usable device memory is ~94.4 GiB after the CUDA context).

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<trainer GPU UUID> \
        ~/envs/train/bin/python tests/trainer/bench_train_step.py \
        [--no-liger] [--lora-fp32] [--lengths 8192,16384] [--offload-lengths 49152,65536] [--out results.json]
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from rltldr import model_utils as mu  # noqa: E402
from rltldr.config import load_config  # noqa: E402

BASE_DIR = load_config().trainer_base_dir      # dequantized bf16 checkpoint (tools/dequant_fp8.py)
GiB = 2 ** 30


def micro_step(pm, L, offload, chunk, gen):
    ids = torch.randint(0, 248000, (L,), device="cuda", generator=gen)
    pos = torch.arange(1, L, device="cuda")
    w = torch.randn(L - 1, device="cuda", generator=gen)                 # per-token weights (advantage * IS)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    lp = mu.token_logprobs(pm, ids, pos, chunk=chunk, offload=offload)
    torch.cuda.synchronize()
    t_fwd = time.time() - t0
    loss = -(w * lp).sum() / (L - 1)
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    finite = all(torch.isfinite(p.grad).all().item() for p in pm.parameters() if p.grad is not None)
    pm.zero_grad(set_to_none=False)    # keep grad buffers allocated, as in a real accumulation loop
    return {"L": L, "offload": offload, "s": dt, "fwd_s": t_fwd, "tok_per_s": L / dt,
            "peak_alloc_GiB": torch.cuda.max_memory_allocated() / GiB,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / GiB, "grads_finite": finite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="8192,16384,24576,32768,36864,40960,45056")
    ap.add_argument("--offload-lengths", default="49152,57344,65536,73728,81920,90112")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--lora-fp32", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--nograd-lengths", default="32768,65536")
    ap.add_argument("--out", default=os.path.join(load_config().data, "bench", "bench_train_step.json"))
    args = ap.parse_args()

    pm, _ = mu.load_policy(BASE_DIR, use_liger=not args.no_liger,
                           lora_compute_dtype=torch.float32 if args.lora_fp32 else torch.bfloat16)
    lora = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora, lr=1e-5, betas=(0.9, 0.99), weight_decay=0.0, fused=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    # one tiny step so AdamW state (2 x 0.81 GiB fp32) and grad buffers exist during the measurements
    lp = mu.token_logprobs(pm, torch.randint(0, 1000, (512,), device="cuda"), torch.arange(1, 512, device="cuda"))
    (-lp.mean()).backward()
    opt.step()
    pm.zero_grad(set_to_none=False)
    static = torch.cuda.memory_allocated() / GiB
    free, total = torch.cuda.mem_get_info()
    res = {"liger": not args.no_liger, "lora_compute": "fp32" if args.lora_fp32 else "bf16", "chunk": args.chunk,
           "static_alloc_GiB": static, "device_total_GiB": total / GiB, "steps": [], "nograd": []}
    print(json.dumps({k: v for k, v in res.items() if k not in ("steps", "nograd")}), flush=True)

    for mode, lengths in ((False, args.lengths), (True, args.offload_lengths)):
        for L in [int(x) for x in lengths.split(",") if x]:
            rec = None
            try:
                for rep in range(args.repeats):
                    r = micro_step(pm, L, mode, args.chunk, gen)
                    r["first_call_s"] = rec["first_call_s"] if rec else r["s"]
                    rec = r
                print(json.dumps(rec), flush=True)
                res["steps"].append(rec)
            except torch.OutOfMemoryError as e:
                print(json.dumps({"L": L, "offload": mode, "OOM": str(e).splitlines()[0][:200]}), flush=True)
                res["steps"].append({"L": L, "offload": mode, "OOM": True})
                pm.zero_grad(set_to_none=False)
                gc.collect()
                torch.cuda.empty_cache()
                break
            gc.collect()
            torch.cuda.empty_cache()

    # no-grad forward (old-policy / behaviour log-prob recompute)
    for L in [int(x) for x in args.nograd_lengths.split(",") if x]:
        ids = torch.randint(0, 248000, (L,), device="cuda", generator=gen)
        try:
            with torch.no_grad():
                for _ in range(2):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    mu.token_logprobs(pm, ids, torch.arange(1, L, device="cuda"))
                    torch.cuda.synchronize()
            rec = {"L": L, "s": time.time() - t0, "peak_alloc_GiB": torch.cuda.max_memory_allocated() / GiB}
        except torch.OutOfMemoryError:
            rec = {"L": L, "OOM": True}
        print("nograd", json.dumps(rec), flush=True)
        res["nograd"].append(rec)
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
