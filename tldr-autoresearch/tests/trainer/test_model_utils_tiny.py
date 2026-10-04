"""Fast functional tests of rltldr.model_utils on a tiny random Qwen3.5 composite checkpoint (GPU, ~1 min).

Builds a 4-layer Qwen3_5ForConditionalGeneration (vision tower + MTP keys included, like the real checkpoint),
saves it sharded with the real tokenizer files, then exercises every public function.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<trainer GPU UUID> \
        ~/envs/train/bin/python tests/trainer/test_model_utils_tiny.py
(reads config.json and the tokenizer files of the real checkpoint, config trainer_base_dir)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from rltldr import model_utils as mu  # noqa: E402
from rltldr.config import load_config  # noqa: E402

REAL = load_config().trainer_base_dir
RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def build_tiny_checkpoint(d: Path):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    full = json.loads((Path(REAL) / "config.json").read_text())
    tc = full["text_config"]
    nl = 4
    tc.update(num_hidden_layers=nl, layer_types=tc["layer_types"][:nl], hidden_size=256, intermediate_size=512,
              num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64, vocab_size=4096)
    vc = full["vision_config"]
    vc.update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256)
    cfg = Qwen3_5Config(**{k: v for k, v in full.items() if k not in ("model_type", "architectures",
                                                                       "transformers_version")})
    torch.manual_seed(0)
    m = mq.Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
    with torch.no_grad():   # make norms / A_log non-trivial so a mis-load would show
        for n, p in m.named_parameters():
            if "norm" in n:
                p.normal_(0, 0.1)
    m.save_pretrained(d, max_shard_size="2MB")
    sd = {}
    # add fake MTP tensors like the real checkpoint (must be ignored by the CausalLM loader)
    sd["mtp.fc.weight"] = torch.zeros(256, 512, dtype=torch.bfloat16)
    save_file(sd, str(d / "mtp.safetensors"), metadata={"format": "pt"})
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    idx["weight_map"]["mtp.fc.weight"] = "mtp.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        shutil.copy(Path(REAL) / fn, d / fn)


def ref_logprobs(pm, ids, pos):
    """Reference: full logits through the HF CausalLM forward + log_softmax."""
    with mu.autocast_ctx(pm):
        logits = pm(input_ids=ids[None]).logits[0].float()
    return torch.log_softmax(logits, -1)[pos - 1].gather(-1, ids[pos][:, None]).squeeze(-1)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tiny_qwen35_", dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        build_tiny_checkpoint(tmp)
        n_vis = sum(1 for k in json.loads((tmp / "model.safetensors.index.json").read_text())["weight_map"]
                    if k.startswith("model.visual."))
        pm, tok = mu.load_policy(str(tmp), lora_r=8, lora_alpha=16, verify_weights="full")
        check("load: tokenizer", tok("hello")["input_ids"] and tok.eos_token_id is not None)
        check("load: skipped visual tensors exist in ckpt", n_vis > 0, f"({n_vis} visual)")
        check("load: liger modules active", type(pm.get_base_model().model.norm).__name__.startswith("Liger"))
        torch.manual_seed(1)
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)
        dev = next(pm.parameters()).device
        L = 700
        ids = torch.randint(0, 4096, (L,), device=dev)
        pos = torch.arange(1, L, device=dev)

        # 1. chunked log-probs vs the HF full-logits path and vs an all-fp32 copy of the model ("truth")
        lora = [p for p in pm.parameters() if p.requires_grad]
        names = [n for n, p in pm.named_parameters() if p.requires_grad]

        def grads_of(fn):
            out = fn()
            out.sum().backward()
            g = [p.grad.clone() for p in lora]
            pm.zero_grad(set_to_none=True)
            return out.detach(), g

        def rel(a, b):
            return max(((x - y).norm() / (y.norm() + 1e-12)).item() for x, y in zip(a, b))

        lp, g1 = grads_of(lambda: mu.token_logprobs(pm, ids, pos, chunk=128))
        _, g1b = grads_of(lambda: mu.token_logprobs(pm, ids, pos, chunk=128))
        noise = rel(g1b, g1)
        ref, g2 = grads_of(lambda: ref_logprobs(pm, ids, pos))
        import copy
        from peft.tuners.lora import LoraLayer
        pm32 = copy.deepcopy(pm).float()
        for m in pm32.modules():
            if isinstance(m, LoraLayer):
                m.cast_input_dtype_enabled = True
        pm32.rltldr_cfg = dict(pm.rltldr_cfg, lora_compute_dtype=torch.float32)
        logits32 = pm32(input_ids=ids[None]).logits[0].float()
        truth = torch.log_softmax(logits32, -1)[pos - 1].gather(-1, ids[pos][:, None]).squeeze(-1)
        truth.sum().backward()
        g32 = [p.grad.clone() for n, p in pm32.named_parameters() if n in set(names)]
        del pm32, logits32
        check("chunked logprob values vs HF path", (lp - ref).abs().max().item() < 0.05,
              f"max|d|={(lp - ref).abs().max().item():.2e} (HF path rounds logits to bf16)")
        e_ours, e_hf = rel(g1, g32), rel(g2, g32)
        check("LoRA grads vs fp32 truth: ours no worse than HF bf16 path", e_ours <= 1.5 * e_hf + 1e-3,
              f"ours {e_ours:.2e} | HF full-logits path {e_hf:.2e} | ours-vs-HF {rel(g1, g2):.2e} | "
              f"run-to-run noise {noise:.2e} | logp max|ours-truth|={(lp - truth.detach()).abs().max().item():.2e}")

        # exactness of the vocabulary-chunked function vs fp32 logits from the same hidden states
        with torch.no_grad():
            with mu.autocast_ctx(pm):
                h = pm.get_base_model().model(input_ids=ids[None]).last_hidden_state[0]
                lp_ng = mu.token_logprobs(pm, ids, pos, chunk=100)
            W = pm.get_base_model().lm_head.weight
            ref32 = torch.log_softmax(h.float() @ W.float().t(), -1)[pos - 1].gather(-1, ids[pos][:, None])[:, 0]
        check("chunked logprob vs fp32-logit reference (same hidden)", (lp_ng - ref32).abs().max().item() < 1e-4,
              f"max|d|={(lp_ng - ref32).abs().max().item():.2e}")

        # hidden-state gradient of the custom Function vs autograd through log_softmax (fp32)
        hs = h[:300].detach().float().to(torch.bfloat16).requires_grad_(True)
        tgt = ids[1:301]
        out = mu._ChunkedTokenLogProbs.apply(hs, W, tgt, 64)
        gvec = torch.randn_like(out)
        (out * gvec).sum().backward()
        hs2 = hs.detach().float().requires_grad_(True)
        r = torch.log_softmax(hs2 @ W.float().t(), -1).gather(-1, tgt[:, None])[:, 0]
        (r * gvec).sum().backward()
        herr = ((hs.grad.float() - hs2.grad).norm() / hs2.grad.norm()).item()
        check("custom Function d/dhidden", herr < 1e-2, f"rel err={herr:.2e}")

        # 2. subset / unordered positions and padding/trimming give the same numbers as the full pass
        sub = torch.tensor([5, 650, 17, 17, 300], device=dev)
        with torch.no_grad():
            lp_sub = mu.token_logprobs(pm, ids, sub)
        dsub = (lp_sub - lp_ng[sub - 1]).abs().max().item()
        check("subset positions (trim+pad)", dsub < 2e-2, f"max|d|={dsub:.2e}")

        # 3. autocast does not change the base model; bf16 vs fp32 LoRA compute agree
        with torch.no_grad(), pm.disable_adapter():
            base_ac = mu.token_logprobs(pm, ids, pos)
            h_noac = pm.get_base_model().model(input_ids=ids[None]).last_hidden_state[0]
        base_noac = mu._ChunkedTokenLogProbs.apply(h_noac.index_select(0, pos - 1), W, ids[pos], 2048)
        check("autocast leaves base model unchanged", torch.equal(base_ac, base_noac),
              f"max|d|={(base_ac - base_noac).abs().max().item():.2e}")
        check("adapter changes logprobs", (lp_ng - base_ac).abs().mean().item() > 1e-3,
              f"mean|adapter-base|={(lp_ng - base_ac).abs().mean().item():.3e}")
        from peft.tuners.lora import LoraLayer
        lmods = [m for m in pm.modules() if isinstance(m, LoraLayer)]
        for m in lmods:
            m.cast_input_dtype_enabled = True
        pm.rltldr_cfg["lora_compute_dtype"] = torch.float32
        with torch.no_grad():
            lp32 = mu.token_logprobs(pm, ids, pos)
        for m in lmods:
            m.cast_input_dtype_enabled = False
        pm.rltldr_cfg["lora_compute_dtype"] = torch.bfloat16
        check("bf16 vs fp32 LoRA compute", (lp32 - lp_ng).abs().max().item() < 5e-2,
              f"max|d|={(lp32 - lp_ng).abs().max().item():.2e} mean={(lp32 - lp_ng).abs().mean().item():.2e}")

        # 4. activation offload: same values and grads
        lp_off, g3 = grads_of(lambda: mu.token_logprobs(pm, ids, pos, chunk=128, offload=True))
        goff = rel(g3, g1)
        check("offload values", torch.equal(lp_off, lp), f"max|d|={(lp_off - lp).abs().max().item():.2e}")
        check("offload grads (within run-to-run kernel noise)", goff <= max(3 * noise, 5e-3),
              f"max rel diff={goff:.2e}, run-to-run noise={noise:.2e}")

        # 5. export -> files/keys/config; load back into default and into a second adapter
        out = tmp / "adapters" / "v1"
        mu.export_adapter(pm, out)
        keys = list(safe_open(str(out / mu.ADAPTER_WEIGHTS), "pt").keys())
        cfg = json.loads((out / mu.ADAPTER_CONFIG).read_text())
        sd = load_file(str(out / mu.ADAPTER_WEIGHTS))
        check("export: key prefix", all(k.startswith(mu.VLLM_LAYER_PREFIX) for k in keys), keys[0])
        check("export: bf16", all(t.dtype == torch.bfloat16 for t in sd.values()))
        check("export: config", cfg["r"] == 8 and cfg["lora_alpha"] == 16 and cfg["peft_type"] == "LORA"
              and isinstance(cfg["target_modules"], list), str({k: cfg[k] for k in ("r", "lora_alpha")}))
        check("export: no tmp leftovers", sorted(p.name for p in out.parent.iterdir()) == ["v1"])
        try:
            mu.export_adapter(pm, out)
            check("export: refuses overwrite", False)
        except FileExistsError:
            check("export: refuses overwrite", True)

        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_" in n:
                    p.zero_()
        mu.load_adapter_weights(pm, out)
        with torch.no_grad():
            lp_rt = mu.token_logprobs(pm, ids, pos)
        check("round-trip into default: identical logprobs", torch.equal(lp_rt, lp_ng),
              f"max|d|={(lp_rt - lp_ng).abs().max().item():.2e}")
        # modify default, load v1 as "old" and compare under adapter_context
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B.default" in n:
                    p.mul_(-1)
        flags = {n: p.requires_grad for n, p in pm.named_parameters()}
        mu.load_adapter_weights(pm, out, adapter_name="old")
        check("second adapter frozen", all(not p.requires_grad for n, p in pm.named_parameters() if ".old." in n)
              and all(p.requires_grad == flags[n] for n, p in pm.named_parameters() if n in flags))
        with torch.no_grad():
            lp_new = mu.token_logprobs(pm, ids, pos)
            with mu.adapter_context(pm, "old"):
                lp_old = mu.token_logprobs(pm, ids, pos)
            lp_new2 = mu.token_logprobs(pm, ids, pos)
        check("adapter_context(old) reproduces exported policy", torch.equal(lp_old, lp_ng),
              f"max|d|={(lp_old - lp_ng).abs().max().item():.2e}")
        check("adapter_context restores default", torch.equal(lp_new, lp_new2) and
              (lp_new - lp_ng).abs().mean().item() > 1e-3 and
              all(p.requires_grad == flags[n] for n, p in pm.named_parameters() if n in flags))
        pm.delete_adapter("old")
        check("delete_adapter(old)", not any(".old." in n for n, _ in pm.named_parameters()))

        # 6. trainer state round trip (weights + AdamW state), then identical next steps
        lora = [p for p in pm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(lora, lr=1e-3, fused=True)

        def step(replay_grads=None):
            """One AdamW step; returns (loss, grads). The backward is not bit-deterministic (FLA/SDPA kernels
            use atomics: ~1e-3 relative run-to-run), so a bit-exact resume check replays recorded gradients."""
            opt.zero_grad(set_to_none=True)
            l = -mu.token_logprobs(pm, ids, pos).mean()
            l.backward()
            grads = [p.grad.clone() for p in lora]
            if replay_grads is not None:
                for p, g in zip(lora, replay_grads):
                    p.grad.copy_(g)
            opt.step()
            return l.item(), grads

        def opt_state_snapshot():
            return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()} for p in lora]

        def opt_state_equal(snap_):
            return all(set(opt.state[p]) == set(s) and all(torch.equal(opt.state[p][k], s[k]) for k in s)
                       for p, s in zip(lora, snap_))

        step()
        ck = tmp / "ckpt" / "step1"
        mu.save_trainer_state(pm, opt, ck, metadata={"version": 1, "step": 1}, extra={"note": "x"})
        snap = {n: p.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
        snap_opt = opt_state_snapshot()
        l_a, grads_a = step()
        after_a = {n: p.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
        meta, extra = mu.load_trainer_state(pm, opt, ck)
        same = all(torch.equal(p, snap[n]) for n, p in pm.named_parameters() if p.requires_grad)
        same_opt = opt_state_equal(snap_opt)
        l_b, _ = step(replay_grads=grads_a)
        after_b = {n: p.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
        same_next = all(torch.equal(after_a[n], after_b[n]) for n in after_a)
        check("trainer state: weights + AdamW state restored", same and same_opt
              and meta == {"version": 1, "step": 1} and extra == {"note": "x"})
        check("trainer state: identical next step (same grads)", same_next and l_a == l_b,
              f"loss {l_a:.6f} vs {l_b:.6f}")
        # rolling checkpoint: saving again to the same dir atomically replaces it; perturb the live weights and
        # optimizer first so the load has to restore something
        snap_opt_b = opt_state_snapshot()
        mu.save_trainer_state(pm, opt, ck, metadata={"version": 2, "step": 2})
        step()
        meta2, _ = mu.load_trainer_state(pm, opt, ck)
        now = {n: p.detach() for n, p in pm.named_parameters() if p.requires_grad}
        check("trainer state: atomic overwrite", meta2 == {"version": 2, "step": 2}
              and all(torch.equal(now[n], after_b[n]) for n in now) and opt_state_equal(snap_opt_b)
              and sorted(p.name for p in ck.parent.iterdir()) == ["step1"])
        # weights-only checkpoint: `extra` must survive without an optimizer; asking for optimizer state fails loudly
        ck2 = tmp / "ckpt" / "noopt"
        mu.save_trainer_state(pm, None, ck2, metadata={"version": 3}, extra={"rng": 7})
        meta3, extra3 = mu.load_trainer_state(pm, None, ck2)
        try:
            mu.load_trainer_state(pm, opt, ck2)
            refused = False
        except ValueError:
            refused = True
        check("trainer state: extra kept without optimizer", meta3 == {"version": 3} and extra3 == {"rng": 7}
              and refused)
        mu.export_adapter(pm, out, overwrite=True)
        check("export: overwrite=True replaces atomically", sorted(p.name for p in out.parent.iterdir()) == ["v1"]
              and torch.equal(load_file(str(out / mu.ADAPTER_WEIGHTS))[keys[0]],
                              dict(mu._lora_params(pm, "default"))[keys[0]].detach().cpu().to(torch.bfloat16)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(not v for v in RESULTS.values())
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
