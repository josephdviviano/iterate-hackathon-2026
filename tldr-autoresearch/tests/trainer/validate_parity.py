"""Full-model validation of rltldr.model_utils against the live vLLM server (GPU 3 only).

  (a) load_policy on the real 27B checkpoint: time, GPU memory, key report, bit-exact check of all tensors.
  (b) trainer-vs-vLLM log-prob parity, base and LoRA adapter (random B), on a fixed ~1.5k-token chat:
      prompt rendered with the tokenizer's chat template (checked == vLLM's own rendering) + a completion sampled
      from the served base model. vLLM log-probs come from /v1/completions prompt_logprobs.
  (d) export_adapter -> load_adapter_weights round trip (into "default" and into a second adapter slot).
Also checks that torch.autocast (needed for bf16 LoRA compute) leaves base-model log-probs unchanged.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<trainer GPU UUID> \
        ~/envs/train/bin/python tests/trainer/validate_parity.py [--no-liger]
Writes data/parity/{parity_seq.json, vllm_logprobs_*.json, results*.json}; unloads its adapters from vLLM.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root

import requests  # noqa: E402
import torch  # noqa: E402

from rltldr import model_utils as mu  # noqa: E402
from rltldr.config import load_config  # noqa: E402

_CFG = load_config()
BASE_DIR = _CFG.trainer_base_dir      # dequantized bf16 checkpoint (tools/dequant_fp8.py)
VLLM = _CFG.vllm_url
SERVED = _CFG.base_model_name
PARITY = Path(_CFG.data) / "parity"
ADAPTERS = Path(_CFG.data) / "adapters"

MESSAGES = [
    {"role": "system", "content": "You are an expert ML engineer helping to speed up a small GPT pretraining run."},
    {"role": "user", "content": (
        "Our nanochat-style training script trains a 12-layer GPT on FineWeb-Edu for a fixed 5-minute budget and "
        "reports val_bpb. We use Muon for matrix parameters and AdamW for embeddings, bf16 autocast, "
        "torch.compile, and a cosine warmdown. Propose three concrete, low-risk changes that could lower val_bpb "
        "within the same wall-clock budget. For each, explain the mechanism, the exact code change, and how you "
        "would tell from the logs whether it helped.")},
]


def get_parity_sequence(tok):
    """Fixed sequence (cached on disk): chat-template prompt + base-model sample from vLLM."""
    PARITY.mkdir(parents=True, exist_ok=True)
    f = PARITY / "parity_seq.json"
    if f.exists():
        return json.loads(f.read_text())
    prompt_ids = tok.apply_chat_template(MESSAGES, tokenize=True, add_generation_prompt=True, return_dict=False,
                                         reasoning_effort="low")
    r = requests.post(f"{VLLM}/v1/chat/completions", json=dict(
        model=SERVED, messages=MESSAGES, max_tokens=1400, temperature=1.0, top_p=1.0, top_k=-1, seed=1234,
        return_token_ids=True, chat_template_kwargs={"reasoning_effort": "low"}), timeout=900)
    r.raise_for_status()
    d = r.json()
    vllm_prompt = d["prompt_token_ids"]
    # the trainer always uses the server's ids; record whether our own template rendering agrees
    seq = {"prompt_ids": vllm_prompt, "completion_ids": d["choices"][0]["token_ids"],
           "trainer_template_matches_vllm": vllm_prompt == list(prompt_ids),
           "finish_reason": d["choices"][0]["finish_reason"]}
    f.write_text(json.dumps(seq))
    return seq


def vllm_prompt_logprobs(model_name, ids):
    t0 = time.time()
    r = requests.post(f"{VLLM}/v1/completions", json={"model": model_name, "prompt": ids, "max_tokens": 1,
                                                      "prompt_logprobs": 0, "temperature": 0.0}, timeout=900)
    r.raise_for_status()
    pl = r.json()["choices"][0]["prompt_logprobs"]
    assert pl[0] is None and len(pl) == len(ids)
    out = []
    for t in range(1, len(ids)):
        ent = pl[t][str(ids[t])]
        out.append(ent["logprob"])
    return torch.tensor(out, dtype=torch.float64), time.time() - t0


def stats(a, b):
    d = (a - b).abs()
    return {"mean_abs": d.mean().item(), "median_abs": d.median().item(), "p99_abs": d.quantile(0.99).item(),
            "max_abs": d.max().item(), "mean_signed": (a - b).mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--b-std", type=float, default=0.01)
    ap.add_argument("--skip-full-verify", action="store_true")
    args = ap.parse_args()
    tag = "noliger" if args.no_liger else "liger"
    res = {"liger": not args.no_liger}
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)              # PEFT's lora_A init draws from the global RNG
    print("GPU:", torch.cuda.get_device_name(0), os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    # ---------------- (a) load ----------------
    t0 = time.time()
    pm, tok = mu.load_policy(BASE_DIR, use_liger=not args.no_liger, verify_weights="spot")
    torch.cuda.synchronize()
    res["load_s"] = time.time() - t0
    res["mem_after_load_GiB"] = torch.cuda.memory_allocated() / 2**30
    res["reserved_after_load_GiB"] = torch.cuda.memory_reserved() / 2**30
    free, total = torch.cuda.mem_get_info()
    res["device_used_after_load_GiB"] = (total - free) / 2**30
    if not args.skip_full_verify:
        t0 = time.time()
        res["full_verify_tensors"] = mu.verify_base_weights(pm, BASE_DIR)
        res["full_verify_s"] = time.time() - t0
    print(json.dumps(res, indent=1), flush=True)

    # ---------------- (b) parity ----------------
    seq = get_parity_sequence(tok)
    ids_list = seq["prompt_ids"] + seq["completion_ids"]
    n_prompt = len(seq["prompt_ids"])
    res.update(seq_len=len(ids_list), n_prompt=n_prompt, template_matches_vllm=seq["trainer_template_matches_vllm"])
    ids = torch.tensor(ids_list, device=dev)
    pos = torch.arange(1, len(ids_list), device=dev)
    comp = pos >= n_prompt          # positions predicting completion tokens

    vb_file = PARITY / "vllm_logprobs_base.json"
    if vb_file.exists():
        vllm_base = torch.tensor(json.loads(vb_file.read_text()), dtype=torch.float64)
    else:
        vllm_base, dt = vllm_prompt_logprobs(SERVED, ids_list)
        vb_file.write_text(json.dumps(vllm_base.tolist()))
        res["vllm_base_request_s"] = dt
    vllm_base_again, _ = vllm_prompt_logprobs(SERVED, ids_list)
    res["vllm_base_rerun_max_abs"] = (vllm_base_again - vllm_base).abs().max().item()

    with torch.no_grad(), pm.disable_adapter():
        t0 = time.time()
        tr_base = mu.token_logprobs(pm, ids, pos).double().cpu()
        res["trainer_nograd_forward_s"] = time.time() - t0
        tr_base2 = mu.token_logprobs(pm, ids, pos).double().cpu()
        # without autocast: identical numerics expected for the base model
        bb, W = pm.get_base_model().model, pm.get_base_model().lm_head.weight
        pad = (-(len(ids_list) - 1)) % 64
        x = torch.cat([ids[:-1], ids.new_zeros(pad)])
        h = bb(input_ids=x[None]).last_hidden_state[0].index_select(0, pos - 1)
        tr_base_noac = mu._ChunkedTokenLogProbs.apply(h, W, ids[pos], 2048).double().cpu()
        del h
    res["trainer_base_rerun_max_abs"] = (tr_base2 - tr_base).abs().max().item()
    res["autocast_vs_no_autocast_base_max_abs"] = (tr_base_noac - tr_base).abs().max().item()
    res["base_trainer_vs_vllm"] = stats(tr_base, vllm_base)
    res["base_trainer_vs_vllm_completion_only"] = stats(tr_base[comp.cpu()], vllm_base[comp.cpu()])
    res["base_trainer_vs_vllm_prompt_only"] = stats(tr_base[~comp.cpu()], vllm_base[~comp.cpu()])
    res["mean_logp_vllm_base_completion"] = vllm_base[comp.cpu()].mean().item()
    res["mean_logp_vllm_base_prompt"] = vllm_base[~comp.cpu()].mean().item()
    arrays = {"trainer_base": tr_base.tolist()}
    print(json.dumps({k: res[k] for k in ("base_trainer_vs_vllm", "autocast_vs_no_autocast_base_max_abs",
                                          "vllm_base_rerun_max_abs", "trainer_base_rerun_max_abs")}, indent=1),
          flush=True)

    if args.no_liger:   # liger comparison run: base parity only
        (PARITY / f"trainer_logprobs_{tag}.json").write_text(json.dumps(arrays))
        (PARITY / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    # random adapter: B ~ N(0, b_std), A = PEFT's kaiming-uniform init
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g, device=dev) * args.b_std)
    with torch.no_grad():
        tr_ad = mu.token_logprobs(pm, ids, pos).double().cpu()
    existing = [p.name for p in ADAPTERS.glob("parity-test-*")]
    n = 1 + max([int(x.rsplit("-", 1)[1]) for x in existing if x.rsplit("-", 1)[1].isdigit()] or [0])
    name = f"parity-test-{n}"
    t0 = time.time()
    path = mu.export_adapter(pm, ADAPTERS / name)
    res["export_s"] = time.time() - t0
    res["adapter_bytes"] = sum(f.stat().st_size for f in Path(path).iterdir())
    vllm_name = f"{name}-{int(time.time())}"
    t0 = time.time()
    r = requests.post(f"{VLLM}/v1/load_lora_adapter", json={"lora_name": vllm_name, "lora_path": path}, timeout=600)
    res["vllm_load_status"] = r.status_code
    res["vllm_load_s"] = time.time() - t0
    r.raise_for_status()
    try:
        vllm_ad, dt = vllm_prompt_logprobs(vllm_name, ids_list)
        res["vllm_adapter_first_request_s"] = dt
    finally:
        u = requests.post(f"{VLLM}/v1/unload_lora_adapter", json={"lora_name": vllm_name}, timeout=120)
        res["vllm_unload_status"] = u.status_code
    (PARITY / "vllm_logprobs_adapter.json").write_text(json.dumps(vllm_ad.tolist()))

    eff_tr, eff_vl = tr_ad - tr_base, vllm_ad - vllm_base
    res["adapter_name_vllm"] = vllm_name
    res["adapter_path"] = path
    res["adapter_trainer_vs_vllm"] = stats(tr_ad, vllm_ad)
    res["adapter_trainer_vs_vllm_completion_only"] = stats(tr_ad[comp.cpu()], vllm_ad[comp.cpu()])
    res["adapter_effect_trainer_mean_abs"] = eff_tr.abs().mean().item()
    res["adapter_effect_vllm_mean_abs"] = eff_vl.abs().mean().item()
    res["adapter_effect_vllm_identical_to_base"] = bool(torch.equal(vllm_ad, vllm_base))
    res["adapter_effect_corr"] = torch.corrcoef(torch.stack([eff_tr, eff_vl]))[0, 1].item()
    res["adapter_effect_diff"] = stats(eff_tr, eff_vl)
    c = comp.cpu()
    res["completion_only"] = {
        "adapter_effect_trainer_mean_abs": eff_tr[c].abs().mean().item(),
        "adapter_effect_vllm_mean_abs": eff_vl[c].abs().mean().item(),
        "adapter_effect_corr": torch.corrcoef(torch.stack([eff_tr[c], eff_vl[c]]))[0, 1].item(),
        "adapter_effect_diff": stats(eff_tr[c], eff_vl[c])}
    arrays["trainer_adapter"] = tr_ad.tolist()
    (PARITY / f"trainer_logprobs_{tag}.json").write_text(json.dumps(arrays))
    res["mean_logp_trainer_adapter_completion"] = tr_ad[comp.cpu()].mean().item()
    res["mean_logp_vllm_adapter_completion"] = vllm_ad[comp.cpu()].mean().item()

    # ---------------- (d) round trip ----------------
    with torch.no_grad():
        for n_, p in pm.named_parameters():
            if "lora_" in n_:
                p.zero_()
        t0 = time.time()
        mu.load_adapter_weights(pm, path)
        res["load_adapter_s"] = time.time() - t0
        tr_rt = mu.token_logprobs(pm, ids, pos).double().cpu()
        for n_, p in pm.named_parameters():
            if "lora_B.default" in n_:
                p.zero_()
        mu.load_adapter_weights(pm, path, adapter_name="old")
        with mu.adapter_context(pm, "old"):
            tr_old = mu.token_logprobs(pm, ids, pos).double().cpu()
        tr_after = mu.token_logprobs(pm, ids, pos).double().cpu()
    res["roundtrip_default_identical"] = bool(torch.equal(tr_rt, tr_ad))
    res["roundtrip_default_max_abs"] = (tr_rt - tr_ad).abs().max().item()
    res["roundtrip_second_adapter_identical"] = bool(torch.equal(tr_old, tr_ad))
    res["after_context_default_is_zero_B_equals_base"] = (tr_after - tr_base).abs().max().item()
    res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    (PARITY / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
