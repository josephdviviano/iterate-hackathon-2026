"""Per-module LoRA transfer check against the live vLLM server.

For each target module type (and for all of them together) the trainer fits a LoRA on that module type only
(other B = 0) for a few Adam steps to *raise* the log-likelihood of the parity sequence's completion tokens,
exports it, loads it into vLLM, and compares the resulting shift in log-probs in both systems. A systematic,
trained effect transfers only if vLLM applies every tensor to the right module/rows; a dropped or mis-sliced
module shows up as a vLLM shift near zero (or of a different size). Random-B adapters are a poor probe here:
vLLM's FP8 activation quantisation makes per-token log-probs jump by ~0.1 nat for *any* tiny weight change
(`--mode random --b-std 0.0001` measures that noise floor), so random effects are noise-dominated.

Requires data/parity/{parity_seq.json, vllm_logprobs_base.json} from validate_parity.py.
Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<trainer GPU UUID> \
        ~/envs/train/bin/python tests/trainer/validate_adapter_modules.py
"""
import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                     # tests/trainer

import requests  # noqa: E402
import torch  # noqa: E402

from rltldr import model_utils as mu  # noqa: E402
from validate_parity import ADAPTERS, BASE_DIR, PARITY, VLLM, vllm_prompt_logprobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trained", "random"], default="trained")
    ap.add_argument("--b-std", default="0.02", help="random mode: comma-separated B std values")
    ap.add_argument("--modules", default="ALL," + ",".join(mu.DEFAULT_TARGET_MODULES))
    ap.add_argument("--target-gain", type=float, default=0.15, help="trained mode: stop when mean completion "
                                                                      "log-prob rose by this much (nats/token)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--keep-adapters", action="store_true")
    args = ap.parse_args()
    seq = json.loads((PARITY / "parity_seq.json").read_text())
    ids_list = seq["prompt_ids"] + seq["completion_ids"]
    n_prompt = len(seq["prompt_ids"])
    vllm_base = torch.tensor(json.loads((PARITY / "vllm_logprobs_base.json").read_text()), dtype=torch.float64)
    pm, _ = mu.load_policy(BASE_DIR)
    dev = torch.device("cuda", 0)
    ids = torch.tensor(ids_list, device=dev)
    pos = torch.arange(1, len(ids_list), device=dev)
    comp = (pos >= n_prompt).cpu()
    with torch.no_grad(), pm.disable_adapter():
        tr_base = mu.token_logprobs(pm, ids, pos).double().cpu()
    init = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
    out = {}
    stamp = int(time.time())
    runs = ([(m, float(s)) for m in args.modules.split(",") for s in args.b_std.split(",")]
            if args.mode == "random" else [(m, None) for m in args.modules.split(",")])
    for mod, std in runs:
        sel = (lambda n: "lora_" in n) if mod == "ALL" else (lambda n, m=mod: f".{m}.lora_" in n)
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_" in n:
                    p.copy_(init[n])          # A = PEFT init, B = 0
        steps = 0
        if args.mode == "random":
            g = torch.Generator(device=dev).manual_seed(0)
            with torch.no_grad():
                for n, p in pm.named_parameters():
                    if "lora_B" in n and sel(n):
                        p.copy_(torch.randn(p.shape, generator=g, device=dev) * std)
        else:
            params = [p for n, p in pm.named_parameters() if p.requires_grad and sel(n)]
            opt = torch.optim.Adam(params, lr=args.lr)
            for steps in range(1, args.max_steps + 1):
                opt.zero_grad(set_to_none=True)
                lp = mu.token_logprobs(pm, ids, pos)
                loss = -lp[comp.to(dev)].mean()
                loss.backward()
                opt.step()
                gain = -loss.item() - tr_base[comp].mean().item()
                if gain >= args.target_gain:
                    break
        with torch.no_grad():
            tr = mu.token_logprobs(pm, ids, pos).double().cpu()
        tag = f"{mod}" + (f"@{std:g}" if std is not None else "")
        name = f"modcheck-{tag}-{stamp}"
        path = mu.export_adapter(pm, ADAPTERS / name)
        requests.post(f"{VLLM}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path},
                      timeout=600).raise_for_status()
        try:
            vl, _ = vllm_prompt_logprobs(name, ids_list)
        finally:
            requests.post(f"{VLLM}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
            if not args.keep_adapters:
                shutil.rmtree(path, ignore_errors=True)
        et, ev = tr - tr_base, vl - vllm_base
        out[tag] = {
            "steps": steps,
            "trainer_completion_mean_shift": et[comp].mean().item(),
            "vllm_completion_mean_shift": ev[comp].mean().item(),
            "trainer_prompt_mean_shift": et[~comp].mean().item(),
            "vllm_prompt_mean_shift": ev[~comp].mean().item(),
            "trainer_mean_abs_effect": et.abs().mean().item(),
            "vllm_mean_abs_effect": ev.abs().mean().item(),
            "effect_corr_completion": torch.corrcoef(torch.stack([et[comp], ev[comp]]))[0, 1].item(),
            "effect_corr_all": torch.corrcoef(torch.stack([et, ev]))[0, 1].item(),
            "adapter_mismatch_mean_abs": (tr - vl).abs().mean().item(),
            "adapter_mismatch_completion_mean_abs": (tr - vl)[comp].abs().mean().item()}
        print(tag, json.dumps({k: round(v, 4) for k, v in out[tag].items()}), flush=True)
    (PARITY / f"module_check_{args.mode}_{stamp}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
