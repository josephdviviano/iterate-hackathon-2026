"""Standalone inference for the Qwen3.8-27B layer-42 Natural Language Autoencoder.

Needs only torch, transformers (>=5.5, with qwen3_5), peft, safetensors, huggingface_hub
(+ flash-linear-attention and causal-conv1d for speed). No EasyNLA dependency.

    from nla_qwen38 import NLA
    nla = NLA.load("gereon/qwen3.8-27b-nla-L42", stage="rl")      # or stage="sft"
    acts = nla.extract(["The capital of France is"])               # [N, 5120] layer-42 residuals
    expl = nla.explain(acts)                                        # list[str]
    recon = nla.reconstruct(expl)                                   # [N, 5120]
    print(expl[0], nla.fve(recon, acts))

Memory: verbalizer = full 27B (~54 GB bf16); reconstructor = first 43 blocks (~36 GB).
Pass `device` / `ar_device` to put them on different GPUs.

Conventions (identical to EasyNLA / ceselder's Qwen3.6 NLA):
  * activation = output of decoder block 42 at the LAST token of the text (no BOS added).
  * AV: the activation is ADDED, norm-matched, to the residual stream after block 1 at the
    marker token `㈜` (h <- h + ||h|| * v/||v||; Karvonen et al.), then the model writes
    <explanation>...</explanation>.
  * AR: blocks 0..42 of the base (+LoRA), final norm removed, last-token hidden state
    normalized to sqrt(d) and passed through a Linear(d, d) head.
  * FVE = 1 - MSE(n(pred), n(gold)) / MSE(n(gold), mean n(gold)), n = scale to norm sqrt(d).
"""

import json
import math
import os
import re

import torch

BASE = "Qwen/Qwen3.8-27B"
LAYER = 42
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a",
                "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"]
EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.S)


def load_text_lm(repo=BASE, device="cuda:0", dtype=torch.bfloat16, num_layers=None):
    """Text-only Qwen3_5ForCausalLM from the multimodal checkpoint (vision tower / MTP dropped)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    d = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(d).text_config
    cfg.architectures = ["Qwen3_5ForCausalLM"]
    if num_layers:
        cfg.num_hidden_layers = num_layers
        cfg.layer_types = list(cfg.layer_types)[:num_layers]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    model = model.to_empty(device=device)
    params = dict(model.named_parameters())
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    todo = {}
    for k, f in idx.items():
        if k.startswith("model.language_model."):
            dst = "model." + k[len("model.language_model."):]
        elif k.startswith("lm_head."):
            dst = k
        else:
            continue
        if dst in params:
            todo.setdefault(f, []).append((k, dst))
    filled = set()
    with torch.no_grad():
        for f, pairs in todo.items():
            with safe_open(f"{d}/{f}", "pt", device=str(device)) as h:
                for src, dst in pairs:
                    params[dst].copy_(h.get_tensor(src).to(dtype))
                    filled.add(dst)
    missing = set(params) - filled
    assert not missing, f"unfilled params: {sorted(missing)[:5]}"
    for m in model.modules():   # non-persistent rotary buffers are garbage after to_empty
        if hasattr(m, "rope_init_fn") and hasattr(m, "inv_freq"):
            inv, sc = m.rope_init_fn(m.config, device)
            m.inv_freq.copy_(inv)
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = inv.clone()
            m.attention_scaling = sc
    return model.eval()


class _Critic(torch.nn.Module):
    def __init__(self, backbone, d):
        super().__init__()
        self.backbone = backbone
        self.value_head = torch.nn.Linear(d, d, bias=False)


class NLA:
    def __init__(self, repo_dir, stage="rl", device="cuda:0", ar_device=None, load_av=True, load_ar=True):
        from peft import LoraConfig, PeftModel, inject_adapter_in_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.dir = repo_dir
        self.meta = json.load(open(f"{repo_dir}/nla_config.json"))
        self.device, self.ar_device = device, ar_device or device
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.prompt_ids = self.tok.encode(self.meta["actor_prompt_rendered"], add_special_tokens=False)
        assert self.prompt_ids == self.meta["actor_prompt_ids"], "tokenizer drift: prompt ids differ"
        self.inj = self.meta["injection_token_id"]
        self.d = self.meta["d_model"]
        self.scale = math.sqrt(self.d)
        has_rl = os.path.exists(f"{repo_dir}/av_rl_lora/adapter_model.safetensors")
        stage = stage if (stage == "sft" or has_rl) else "sft"
        self.stage = stage
        self.av = None
        if load_av:
            # LoRAs stay UNMERGED and stacked (sft + rl): extraction runs with adapters disabled,
            # so one copy of the weights serves as both the pristine base and the verbalizer.
            m = PeftModel.from_pretrained(load_text_lm(BASE, device), f"{repo_dir}/av_sft_lora", adapter_name="sft")
            if stage == "rl":
                m.load_adapter(f"{repo_dir}/av_rl_lora", adapter_name="rl")
                m.base_model.set_adapter(["sft", "rl"])
            self.av = m.eval()
            self._vec = None
            self._layers = m.base_model.model.model.layers
            self._layers[1].register_forward_hook(self._inject_hook)
        if load_ar:
            bb = load_text_lm(BASE, self.ar_device, num_layers=LAYER + 1)
            bb.model.norm = torch.nn.Identity()
            bb.lm_head = torch.nn.Identity()
            critic = _Critic(bb, self.d)
            inject_adapter_in_model(LoraConfig(r=self.meta["ar_lora_r"], lora_alpha=16, lora_dropout=0.0,
                                               use_rslora=True, target_modules=LORA_TARGETS), critic.backbone)
            path = (f"{repo_dir}/rl_critic/ar_lora_value_head.safetensors" if stage == "rl"
                    else f"{repo_dir}/ar_sft_critic/ar_lora_value_head.safetensors")
            sd = load_file(path)
            missing, unexpected = critic.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected[:3]
            assert not [k for k in missing if "lora_" in k or "value_head" in k], "AR weights missing"
            critic.value_head.float()
            self.ar = critic.to(self.ar_device).eval()

    @classmethod
    def load(cls, repo_id="gereon/qwen3.8-27b-nla-L42", **kw):
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo_id), **kw)

    # ---- activations -------------------------------------------------------------
    @torch.no_grad()
    def extract(self, texts, layer=LAYER, batch_size=16):
        """Residual stream after block `layer` at the last token of each text (fp32 CPU [N, d])."""
        import contextlib
        if self.av is not None:
            model, layers, ctx = self.av.base_model.model, self._layers, self.av.disable_adapter()
        else:
            if not hasattr(self, "_base"):
                self._base = load_text_lm(BASE, self.device)
            model, layers, ctx = self._base, self._base.model.layers, contextlib.nullcontext()
        out = []
        grab = {}

        class Stop(Exception):
            pass

        def hook(_m, _i, o):
            grab["h"] = o[0] if isinstance(o, tuple) else o
            raise Stop
        h = layers[layer].register_forward_hook(hook)
        ctx.__enter__()
        try:
            for s in range(0, len(texts), batch_size):
                enc = [self.tok.encode(t, add_special_tokens=False) for t in texts[s:s + batch_size]]
                L = max(map(len, enc))
                ids = torch.full((len(enc), L), self.tok.pad_token_id or 0, dtype=torch.long)
                am = torch.zeros_like(ids)
                for i, e in enumerate(enc):
                    ids[i, :len(e)] = torch.tensor(e)
                    am[i, :len(e)] = 1
                try:
                    model.model(input_ids=ids.to(self.device), attention_mask=am.to(self.device), use_cache=False)
                except Stop:
                    pass
                last = am.sum(1) - 1
                out.append(grab["h"][torch.arange(len(enc)), last.to(self.device)].float().cpu())
        finally:
            h.remove()
            ctx.__exit__(None, None, None)
        return torch.cat(out)

    # ---- AV ------------------------------------------------------------------------
    def _inject_hook(self, _m, _i, out):
        if self._vec is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] < 2:          # decode steps
            return out
        pos = len(self.prompt_ids) - 1 - self.prompt_ids[::-1].index(self.inj)
        h = h.clone()
        v = self._vec.to(h.device, torch.float32)
        hp = h[:, pos].float()
        h[:, pos] = (hp + hp.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)).to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    @torch.no_grad()
    def explain(self, acts, max_new_tokens=256, temperature=0.0, batch_size=64, return_raw=False):
        stop = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|endoftext|>")]
        res = []
        for s in range(0, len(acts), batch_size):
            v = acts[s:s + batch_size]
            pt = torch.tensor([self.prompt_ids] * len(v), device=self.device)
            self._vec = v
            try:
                kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 \
                    else dict(do_sample=False)
                o = self.av.generate(input_ids=pt, attention_mask=torch.ones_like(pt), max_new_tokens=max_new_tokens,
                                     eos_token_id=stop, pad_token_id=stop[0], **kw)
            finally:
                self._vec = None
            for r in range(len(v)):
                raw = self.tok.decode(o[r, pt.shape[1]:], skip_special_tokens=True)
                m = EXPL_RE.search(raw)
                res.append(raw if return_raw else (m.group(1).strip() if m else None))
        return res

    # ---- AR ------------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, explanations, batch_size=64):
        tmpl = self.meta["critic_prompt_template"]
        out = torch.full((len(explanations), self.d), float("nan"))
        idx = [i for i, e in enumerate(explanations) if e]
        for s in range(0, len(idx), batch_size):
            j = idx[s:s + batch_size]
            enc = [self.tok.encode(tmpl.format(explanation=explanations[i]), add_special_tokens=False)[-512:]
                   for i in j]
            L = max(map(len, enc))
            ids = torch.zeros((len(enc), L), dtype=torch.long)
            am = torch.zeros_like(ids)
            for r, e in enumerate(enc):
                ids[r, :len(e)] = torch.tensor(e)
                am[r, :len(e)] = 1
            h = self.ar.backbone.model(input_ids=ids.to(self.ar_device), attention_mask=am.to(self.ar_device),
                                       use_cache=False).last_hidden_state
            last = h[torch.arange(len(enc)), (am.sum(1) - 1).to(self.ar_device)].float()
            last = last / last.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.scale
            out[j] = self.ar.value_head(last).float().cpu()
        return out

    def fve(self, pred, gold):
        """Paper-style FVE over a batch (rows with NaN predictions are dropped)."""
        ok = ~torch.isnan(pred).any(-1)
        n = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.scale
        g, p = n(gold.float()), n(pred[ok].float())
        base = ((g - g.mean(0, keepdim=True)) ** 2).mean()
        return float(1 - ((p - g[ok]) ** 2).mean() / base)
