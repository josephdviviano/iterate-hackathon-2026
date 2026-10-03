"""Natural Language Autoencoder (NLA) for Qwen3.8-27B, layer 42 — full pipeline on Modal.

Run from this directory. Long jobs: `modal deploy modal_nla38.py` then `python spawn.py <fn> '<json>'`.

  download / weight_diff / transfer_test   does ceselder's Qwen3.6 NLA transfer to 3.8? (no)
  build_pool                               text+gold-explanation pools (doc-hash disjoint splits)
  extract                                  Qwen3.8 layer-42 last-token activations
  train_av / train_ar                      AV / AR SFT (data parallel over B200s)
  train_rl                                 GRPO with co-trained critic (data parallel, $ cap)
  evaluate                                 held-out FVE + shuffled-activation control
  export / push_hf / push_data             bundle + upload to Hugging Face
"""

import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
# GPU type for every function. Default B200; set NLA_GPU=H200 for workspaces without B200 access
# (Hopper training needs the triton>=3.7.1 image, which train_rl uses; see check_hopper_triton).
GPU = os.environ.get("NLA_GPU", "B200")


def G(n=None):
    return GPU if n is None else f"{GPU}:{n}"

APP_NAME = "qwen38-nla"
VOL = "/vol"
HF_CACHE = f"{VOL}/hf_cache"
BASE36 = "Qwen/Qwen3.6-27B"
BASE38 = "Qwen/Qwen3.8-27B"
NLA36 = "ceselder/qwen3.6-27b-nla-L42"
DATA8B = "ceselder/qwen3-8b-nla-L24-finefineweb-100k"   # texts + Sonnet-4.6 gold explanations
LAYER = 42
EASYNLA_COMMIT = "d23cba0254e835147d8808a83e4b758976aadaff"   # ceselder/EasyNLA main, qwen3_5 support

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.0", "transformers==5.18.0", "accelerate", "peft==0.21.2",
        "flash-linear-attention==0.5.2", "kernels",
        "pyarrow", "pandas", "numpy", "safetensors", "pyyaml", "httpx", "orjson",
        "huggingface_hub[hf_xet]", "sentencepiece", "protobuf", "wandb", "datasets",
    )
    # prebuilt wheel: without it transformers falls back to a slow torch conv1d
    # in all 48 linear-attention layers
    .pip_install("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    .run_commands(
        "git clone https://github.com/ceselder/EasyNLA /root/EasyNLA",
        f"cd /root/EasyNLA && git checkout {EASYNLA_COMMIT}",
    )
    .env({"HF_HOME": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/EasyNLA"})
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("qwen38-nla", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(volumes={VOL: vol}, timeout=2 * 3600, cpu=8.0, memory=32768, secrets=secrets)
def download():
    from huggingface_hub import snapshot_download
    for repo in (BASE36, BASE38):
        p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
        print(repo, "->", p, flush=True)
        vol.commit()
    p = snapshot_download(NLA36, allow_patterns=["nla_meta.yaml", "av_sft_lora/*", "av_rl_lora_step400/*",
                                                 "rl_critic_step400/*"])
    print(NLA36, "->", p, flush=True)
    p = snapshot_download(DATA8B, repo_type="dataset")
    print(DATA8B, "->", p, flush=True)
    vol.commit()


@app.function(volumes={VOL: vol}, timeout=3600, cpu=8.0, memory=65536, secrets=secrets)
def weight_diff():
    """Relative Frobenius change ||W38 - W36|| / ||W36|| per language-model tensor."""
    import json
    import re
    from collections import defaultdict

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    d36, d38 = snapshot_download(BASE36), snapshot_download(BASE38)
    idx36 = json.load(open(f"{d36}/model.safetensors.index.json"))["weight_map"]
    idx38 = json.load(open(f"{d38}/model.safetensors.index.json"))["weight_map"]
    assert set(idx36) == set(idx38), "tensor sets differ"
    by_kind = defaultdict(list)
    by_layer = defaultdict(list)
    handles = {}

    def get(d, f, k):
        key = (d, f)
        if key not in handles:
            handles[key] = safe_open(f"{d}/{f}", "pt")
        return handles[key].get_tensor(k).float()

    for k in sorted(idx36):
        if "visual" in k or k.startswith("mtp"):
            continue
        a, b = get(d36, idx36[k], k), get(d38, idx38[k], k)
        rel = ((b - a).norm() / a.norm().clamp_min(1e-12)).item()
        kind = re.sub(r"layers\.\d+\.", "layers.N.", k)
        by_kind[kind].append(rel)
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            by_layer[int(m.group(1))].append(rel)
        else:
            print(f"  {k:70s} rel_change={rel:.4f}", flush=True)
    print("\n=== mean relative change by tensor kind ===")
    for kind, v in sorted(by_kind.items()):
        print(f"  {kind:75s} {sum(v)/len(v):.4f}  (n={len(v)})")
    print("\n=== mean relative change by layer ===")
    for li in sorted(by_layer):
        v = by_layer[li]
        print(f"  layer {li:2d}: {sum(v)/len(v):.4f}")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_text_lm(repo, dtype=None, device=None):
    """Text-only Qwen3_5ForCausalLM from a multimodal Qwen3.x checkpoint.

    Adapter keys in the 3.6 NLA are `model.layers.N...` (text-only module tree),
    so we instantiate the text model and load `model.language_model.*` weights
    into it by renaming, asserting every parameter was filled.
    """
    import json

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = dtype or torch.bfloat16
    device = device or dev()
    d = snapshot_download(repo)
    tcfg = AutoConfig.from_pretrained(d).text_config
    tcfg.architectures = ["Qwen3_5ForCausalLM"]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tcfg, dtype=dtype)
    model = model.to_empty(device=device)
    want = dict(model.named_parameters())
    want.update(dict(model.named_buffers()))
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    filled = set()
    by_file = {}
    for k, f in idx.items():
        if k.startswith("model.language_model."):
            by_file.setdefault(f, []).append((k, "model." + k[len("model.language_model."):]))
        elif k.startswith("lm_head."):
            by_file.setdefault(f, []).append((k, k))
    with torch.no_grad():
        for f, pairs in by_file.items():
            with safe_open(f"{d}/{f}", "pt", device=device) as h:
                for src, dst in pairs:
                    if dst in want:
                        t = h.get_tensor(src)
                        assert want[dst].shape == t.shape, (dst, want[dst].shape, t.shape)
                        want[dst].copy_(t.to(want[dst].dtype))
                        filled.add(dst)
    params = {n for n, _ in model.named_parameters()}
    missing = params - filled
    assert not missing, f"unfilled params: {sorted(missing)[:10]}"
    # non-persistent buffers (rotary inv_freq) must be recomputed after to_empty
    for m in model.modules():
        if hasattr(m, "rope_init_fn") and hasattr(m, "inv_freq"):
            inv, sc = m.rope_init_fn(m.config, device)
            m.inv_freq.copy_(inv)
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = inv.clone()
            m.attention_scaling = sc
    if getattr(tcfg, "tie_word_embeddings", False):
        model.tie_weights()
    model.eval()
    return model


def dev():
    import torch
    return f"cuda:{torch.cuda.current_device()}"


def nla_tokenizer():
    """Tokenizer + chat template exactly as the 3.6 NLA was trained with."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    d = snapshot_download(NLA36, allow_patterns=["av_sft_lora/*"])
    tok = AutoTokenizer.from_pretrained(f"{d}/av_sft_lora")
    return tok


def heldout_rows(n, permille=5):
    """Rows from the RL split whose doc hash is in the <permille bucket.

    The av/ar/rl splits are doc-disjoint, and any doc-hash val split the 3.6 RL
    run used (permille >= 5) contains this bucket, so the 3.6 NLA never trained
    on these docs.
    """
    import re

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    from nla.val_split import is_val_doc

    d = snapshot_download(DATA8B, repo_type="dataset", allow_patterns=["rl_shuf*"])
    pf = pq.ParquetFile(f"{d}/rl_shuf.parquet")
    cols = [c for c in ("doc_id", "detokenized_text_truncated", "prompt", "response")
            if c in pf.schema_arrow.names]
    print("rl_shuf columns:", pf.schema_arrow.names, flush=True)
    out = []
    for b in pf.iter_batches(batch_size=20000, columns=cols):
        dd = b.to_pydict()
        for i in range(b.num_rows):
            if is_val_doc(dd["doc_id"][i], permille):
                expl = None
                if "response" in dd and dd["response"][i]:
                    m = re.search(r"<explanation>(.*?)</explanation>", dd["response"][i], re.S)
                    expl = (m.group(1) if m else dd["response"][i]).strip()
                elif "prompt" in dd and isinstance(dd["prompt"][i], str):
                    m = re.search(r"<text>(.*?)</text>", dd["prompt"][i], re.S)
                    expl = m.group(1).strip() if m else None
                out.append({"doc_id": dd["doc_id"][i], "text": dd["detokenized_text_truncated"][i],
                            "gold": expl})
                if len(out) >= n:
                    return out
    return out


def extract_last_token(model, tok, texts, layer=LAYER, token_budget=65536):
    """Residual-stream output of block `layer` at the last real token (fp32 [N, d])."""
    import torch

    from nla.utils.arch_adapters import resolve_decoder_layers

    layers = resolve_decoder_layers(model)
    grab = {}

    class _Stop(Exception):
        pass

    def hook(_m, _i, out):
        grab["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    h = layers[layer].register_forward_hook(hook)
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    res = [None] * len(texts)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    i = 0
    while i < len(order):
        n = 1
        while i + n < len(order) and (n + 1) * len(enc[order[i + n]]) <= token_budget:
            n += 1
        chunk = order[i:i + n]
        i += n
        L = max(len(enc[k]) for k in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        am = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, k in enumerate(chunk):   # right padding; causal => last real token unaffected
            ids[r, :len(enc[k])] = torch.tensor(enc[k])
            am[r, :len(enc[k])] = 1
        ids, am = ids.cuda(), am.cuda()
        try:
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=am, use_cache=False)
        except _Stop:
            pass
        last = am.sum(1) - 1
        v = grab["h"][torch.arange(len(chunk), device=ids.device), last].float().cpu()
        for r, k in enumerate(chunk):
            res[k] = v[r]
    h.remove()
    return torch.stack(res)


def build_av(base_model, nla_dir, stage="rl"):
    """3.6 NLA verbalizer on `base_model`: merge av_sft_lora, then add the RL LoRA."""
    from peft import PeftModel

    m = PeftModel.from_pretrained(base_model, f"{nla_dir}/av_sft_lora")
    _check_lora_loaded(m)
    m = m.merge_and_unload()
    if stage == "rl":
        m = PeftModel.from_pretrained(m, f"{nla_dir}/av_rl_lora_step400")
        _check_lora_loaded(m)
    m.eval()
    return m


def _check_lora_loaded(peft_model):
    """Guard against silent key mismatch: every lora_B must be nonzero."""
    n, nz = 0, 0
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            n += 1
            nz += int(p.detach().abs().sum().item() > 0)
    assert n > 0 and nz == n, f"LoRA not loaded: {nz}/{n} lora_B nonzero"
    print(f"  LoRA loaded: {n} lora_B tensors, all nonzero", flush=True)


def generate_explanations(av, tok, cfg, acts, max_new_tokens=256, bs=64, do_sample=False):
    """Inject each activation at the marker (Karvonen, layer-1 output) and generate."""
    import torch

    from nla.schema import EXPLANATION_RE
    from nla.utils import build_prompt_text, register_karvonen_hook

    prompt = [{"role": "user", "content": cfg.actor_prompt_template.replace("{injection_char}", "<INJECT>")}]
    ptxt = build_prompt_text(prompt, cfg.injection_char, tok)
    ids = tok.encode(ptxt, add_special_tokens=False)
    assert ids.count(cfg.injection_token_id) == 1, "marker must be exactly one token"
    vref = [None]
    if not getattr(av, "_nla_hooked", False):
        register_karvonen_hook(av, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                               cfg.injection_right_neighbor_id, layer_idx=1)
        av._nla_hooked = True
        av._nla_vref = vref
    vref = av._nla_vref
    outs = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(acts), bs):
        v = acts[s:s + bs]
        # identical prompts -> no padding needed, injection positions identical
        pt = torch.tensor([ids] * len(v), dtype=torch.long, device=dev())
        vref[0] = v.to(dev())
        try:
            with torch.no_grad():
                o = av.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                max_new_tokens=max_new_tokens, do_sample=do_sample,
                                temperature=1.0 if do_sample else None, top_p=None, top_k=None,
                                pad_token_id=pad)
        finally:
            vref[0] = None
        for r in range(len(v)):
            resp = tok.decode(o[r, pt.shape[1]:], skip_special_tokens=True)
            m = EXPLANATION_RE.search(resp)
            outs.append({"resp": resp, "expl": m.group(1).strip() if m else None})
        print(f"    generated {min(s + bs, len(acts))}/{len(acts)}", flush=True)
    return outs


def critic_scores(critic, tok, cfg, texts, gold, bs=32):
    """Per-row normalized MSE between critic(text) and gold; rows with text None -> None."""
    import torch

    from nla.schema import normalize_activation, resolve_target_scale
    from nla.utils.critic import critic_predict

    sc = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    out = [None] * len(texts)
    valid = [i for i, t in enumerate(texts) if t]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(valid), bs):
        idxs = valid[s:s + bs]
        enc = [tok.encode(cfg.critic_prompt_template.format(explanation=texts[i]), add_special_tokens=False)
               for i in idxs]
        L = max(map(len, enc))
        ids = torch.full((len(idxs), L), pad, dtype=torch.long)
        am = torch.zeros((len(idxs), L), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, :len(e)] = torch.tensor(e)
            am[r, :len(e)] = 1
        with torch.no_grad():
            pred = critic_predict(critic, ids.cuda(), am.cuda(), sc)
        pn = normalize_activation(pred.float(), sc)
        gn = normalize_activation(gold[idxs].float().cuda(), sc)
        mse = ((pn - gn) ** 2).mean(-1).cpu()
        for r, i in enumerate(idxs):
            out[i] = mse[r].item()
    return out


def load_critic(path):
    """NLACriticModel + assert backbone weights really came from the checkpoint."""
    import torch
    from safetensors import safe_open

    from nla.models import NLACriticModel
    critic = NLACriticModel.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    sd = {k: v for k, v in critic.state_dict().items()}
    with safe_open(f"{path}/model.safetensors", "pt") as h:
        for k in ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                  "model.language_model.layers.42.mlp.down_proj.weight"):
            ref = h.get_tensor(k)
            cand = [n for n in sd if n.endswith(k.split("model.language_model.")[1])]
            assert len(cand) == 1, (k, cand[:4])
            assert torch.equal(sd[cand[0]].cpu(), ref), f"critic weight {k} not loaded"
    print(f"  critic loaded OK: {len([n for n in sd if 'layers.' in n and n.endswith('input_layernorm.weight')])} blocks",
          flush=True)
    return critic


def fve(mses, gold, cfg):
    import numpy as np

    from nla.schema import compute_predict_mean_baselines, resolve_target_scale
    sc = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    _, base = compute_predict_mean_baselines(gold, sc)
    v = [m for m in mses if m is not None]
    return (1 - float(np.mean(v)) / base) * 100 if v else float("nan"), len(v) / len(mses)


@app.function(gpu=G(), volumes={VOL: vol}, timeout=3 * 3600, secrets=secrets)
def transfer_test(n: int = 256, max_new_tokens: int = 256):
    import gc
    import json
    import time

    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download

    from nla.config import load_nla_config
    from nla.models import NLACriticModel

    t0 = time.time()
    nla_dir = snapshot_download(NLA36, allow_patterns=["nla_meta.yaml", "av_sft_lora/*",
                                                       "av_rl_lora_step400/*", "rl_critic_step400/*"])
    tok = nla_tokenizer()
    cfg = load_nla_config(f"{nla_dir}/av_sft_lora", tok)
    print(f"cfg: inj={cfg.injection_token_id} d={cfg.d_model} mse_scale={cfg.mse_scale} "
          f"inj_scale={cfg.injection_scale}", flush=True)
    rows = heldout_rows(n)
    texts = [r["text"] for r in rows]
    golds = [r["gold"] for r in rows]
    print(f"{len(rows)} held-out rows, {sum(g is not None for g in golds)} with gold expl", flush=True)
    print("sample gold:", golds[0][:300] if golds[0] else None, flush=True)

    acts, gens = {}, {}
    for tag, repo in (("36", BASE36), ("38", BASE38)):
        print(f"\n=== {repo} ===", flush=True)
        model = load_text_lm(repo)
        acts[tag] = extract_last_token(model, tok, texts)
        print(f"  acts {tuple(acts[tag].shape)} norm p50={acts[tag].norm(dim=-1).median():.1f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        av = build_av(model, nla_dir, "rl")
        gens[tag] = generate_explanations(av, tok, cfg, acts[tag], max_new_tokens=max_new_tokens)
        print(f"  sample expl: {gens[tag][0]['expl']!r}"[:600], flush=True)
        # bonus: the 3.6 AV on 3.8 base reading 3.6 activations? not needed; free memory
        del av, model
        gc.collect()
        torch.cuda.empty_cache()

    a36, a38 = acts["36"], acts["38"]
    cos = F.cosine_similarity(a36, a38, dim=-1)
    nr = a38.norm(dim=-1) / a36.norm(dim=-1)
    # how well does the 3.6 activation itself "explain" the 3.8 activation (normalized-MSE FVE)
    sc = (cfg.d_model ** 0.5)
    n36 = a36 / a36.norm(dim=-1, keepdim=True) * sc
    n38 = a38 / a38.norm(dim=-1, keepdim=True) * sc
    base38 = ((n38 - n38.mean(0, keepdim=True)) ** 2).mean().item()
    fve_cross = (1 - ((n36 - n38) ** 2).mean().item() / base38) * 100
    # centered cosine (removes the shared mean direction, which dominates raw cosine)
    c36, c38 = n36 - n36.mean(0), n38 - n38.mean(0)
    ccos = F.cosine_similarity(c36, c38, dim=-1)
    print(f"\n=== activation drift 3.6 -> 3.8 @ L{LAYER} ===", flush=True)
    print(f"  raw cosine   mean={cos.mean():.4f} p10={cos.quantile(0.1):.4f} min={cos.min():.4f}")
    print(f"  centered cos mean={ccos.mean():.4f} p10={ccos.quantile(0.1):.4f}")
    print(f"  norm ratio   mean={nr.mean():.4f}")
    print(f"  FVE(3.8 acts | predicted by 3.6 acts) = {fve_cross:.1f}%", flush=True)

    critic = load_critic(f"{nla_dir}/rl_critic_step400")
    res = {"n": len(rows), "drift": {"cos": cos.mean().item(), "centered_cos": ccos.mean().item(),
                                     "norm_ratio": nr.mean().item(), "fve_cross": fve_cross}}
    # AR on gold explanations (upper-ish reference; no AV involved)
    for tag in ("36", "38"):
        f, rate = fve(critic_scores(critic, tok, cfg, golds, acts[tag]), acts[tag], cfg)
        res[f"ar_gold_vs_{tag}"] = f
        print(f"AR(gold expl) vs {tag} acts: FVE={f:.1f}% (n_valid={rate:.2f})", flush=True)
    # Full NLA loop
    for tag in ("36", "38"):
        f, rate = fve(critic_scores(critic, tok, cfg, [g["expl"] for g in gens[tag]], acts[tag]),
                      acts[tag], cfg)
        res[f"nla_{tag}"] = f
        res[f"nla_{tag}_extract_rate"] = rate
        print(f"NLA (AV on {tag} base, {tag} acts -> AR): FVE={f:.1f}% extract={rate:.2f}", flush=True)
    # 3.8 explanations scored against 3.6 acts (does the critic or the AV carry the drift?)
    f, _ = fve(critic_scores(critic, tok, cfg, [g["expl"] for g in gens["38"]], acts["36"]), acts["36"], cfg)
    res["nla_38expl_vs_36acts"] = f
    print(f"3.8 NLA explanations scored vs 3.6 acts: FVE={f:.1f}%", flush=True)

    os.makedirs(f"{VOL}/results", exist_ok=True)
    with open(f"{VOL}/results/transfer_test.json", "w") as fh:
        json.dump({"summary": res, "rows": [
            {"text": r["text"][-500:], "gold": r["gold"], "expl36": gens["36"][i]["expl"],
             "expl38": gens["38"][i]["expl"]} for i, r in enumerate(rows)]}, fh, indent=1)
    torch.save(acts, f"{VOL}/results/transfer_acts.pt")
    vol.commit()
    print(json.dumps(res, indent=1))
    print(f"total {time.time()-t0:.0f}s")
    return res




# ===========================================================================
# Standard EasyNLA recipe on Qwen3.8-27B:
#   gold (Sonnet-4.6) explanations -> AV SFT + AR SFT from the 3.8 base -> GRPO
# ===========================================================================
DATA = f"{VOL}/data"
CKPT = f"{VOL}/ckpts"
train_image = image.add_local_file(HERE / "patch_qwen35_gva_518.py", "/root/patch_gva.py", copy=True).run_commands(
    "python /root/patch_gva.py",
)
hopper_image = train_image.pip_install("triton==3.7.1")   # see check_hopper_triton
if GPU != "B200":
    train_image = hopper_image   # Hopper backward needs triton>=3.7.1 everywhere
# NB: B200 only for training. On Hopper, fla (triton<3.7.1) refuses the gated-delta backward
# (wrong results, fla#640); installing tilelang did not unlock it.
@app.function(gpu=G(), timeout=900)
def check_conv1d():
    import torch
    import causal_conv1d
    x = torch.randn(2, 64, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    y = causal_conv1d.causal_conv1d_fn(x, w, None, activation="silu")
    print("causal_conv1d OK", causal_conv1d.__version__, y.shape)


@app.function(volumes={VOL: vol}, timeout=3600, cpu=8.0, memory=65536, secrets=secrets)
def build_pool(n_train: int = 120000, n_val: int = 1024, n_rl: int = 25000, seed: int = 0):
    """Row pools (doc-disjoint by crc32(doc_id) % 1000 bucket):

      bucket < 5      : eval held-out (rl split; used by heldout_rows / evaluate)
      5 <= bucket < 25: val (av/ar splits, gold) -- SFT val + RL eval
      bucket >= 25    : train (av/ar splits, gold) and rl prompts (rl split)
    AV and AR SFT train on the SAME rows.
    """
    import random
    import re
    import zlib

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    d = snapshot_download(DATA8B, repo_type="dataset")
    bucket = lambda doc: zlib.crc32(str(doc).encode()) % 1000
    rows, seen = [], set()
    for split in ("av_sft_shuf", "ar_sft_shuf"):
        pf = pq.ParquetFile(f"{d}/{split}.parquet")
        cols = [c for c in ("doc_id", "detokenized_text_truncated", "prompt", "response")
                if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=20000, columns=cols):
            dd = b.to_pydict()
            for i in range(b.num_rows):
                if split.startswith("av"):
                    m = re.search(r"<explanation>(.*?)</explanation>", dd["response"][i] or "", re.S)
                else:
                    m = re.search(r"<text>(.*?)</text>", dd["prompt"][i] or "", re.S)
                g = m.group(1).strip() if m else None
                t = dd["detokenized_text_truncated"][i]
                if g and t and t not in seen:
                    seen.add(t)
                    rows.append((dd["doc_id"][i], t, g))
        print(split, len(rows), flush=True)
    rl = []
    pf = pq.ParquetFile(f"{d}/rl_shuf.parquet")
    for b in pf.iter_batches(batch_size=20000, columns=["doc_id", "detokenized_text_truncated"]):
        dd = b.to_pydict()
        for i in range(b.num_rows):
            if bucket(dd["doc_id"][i]) >= 25 and dd["detokenized_text_truncated"][i]:
                rl.append((dd["doc_id"][i], dd["detokenized_text_truncated"][i], None))
    tr = [r for r in rows if bucket(r[0]) >= 25]
    va = [r for r in rows if 5 <= bucket(r[0]) < 25]
    rng = random.Random(seed)
    for x in (tr, va, rl):
        rng.shuffle(x)
    pools = {"train": tr[:n_train], "val": va[:n_val], "rl": rl[:n_rl]}
    assert not ({r[0] for r in pools["train"]} & {r[0] for r in pools["val"]})
    os.makedirs(DATA, exist_ok=True)
    for name, rr in pools.items():
        pq.write_table(pa.table({"doc_id": [r[0] for r in rr], "text": [r[1] for r in rr],
                                 "gold": [r[2] for r in rr]}), f"{DATA}/pool_{name}.parquet")
        print(name, len(rr), "rows", len({r[0] for r in rr}), "docs", flush=True)
    vol.commit()


@app.function(volumes={VOL: vol}, gpu=G(), timeout=6 * 3600, secrets=secrets)
def extract(split: str, shard: int = 0, n_shards: int = 1):
    """Qwen3.8 layer-42 residual at the last token of each pool row."""
    import time

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = f"{DATA}/act_{split}_{shard:02d}.parquet"
    if os.path.exists(out_path):
        print("exists:", out_path)
        return
    t0 = time.time()
    tok = nla_tokenizer()
    t = pq.read_table(f"{DATA}/pool_{split}.parquet").to_pydict()
    idx = list(range(shard, len(t["text"]), n_shards))
    texts = [t["text"][i] for i in idx]
    model = load_text_lm(BASE38)
    print(f"{split} shard {shard}: {len(idx)} rows, model loaded [{time.time()-t0:.0f}s]", flush=True)
    a = extract_last_token(model, tok, texts)
    n = a.norm(dim=-1)
    print(f"  extracted [{time.time()-t0:.0f}s] norm p25/50/75={np.percentile(n,25):.1f}/"
          f"{np.percentile(n,50):.1f}/{np.percentile(n,75):.1f}", flush=True)
    pq.write_table(pa.table({
        "doc_id": [t["doc_id"][i] for i in idx], "text": texts, "gold": [t["gold"][i] for i in idx],
        "act38": pa.FixedSizeListArray.from_arrays(pa.array(a.numpy().reshape(-1)), a.shape[1]),
    }), out_path)
    vol.commit()
    print(f"wrote {out_path} [{time.time()-t0:.0f}s]", flush=True)


def _load_acts(split):
    import glob

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    fs = sorted(glob.glob(f"{DATA}/act_{split}_*.parquet"))
    assert fs, f"no activation files for {split}"
    tbl = pa.concat_tables([pq.read_table(f) for f in fs])
    col = tbl.column("act38").combine_chunks()
    acts = torch.from_numpy(np.asarray(col.values.to_numpy(zero_copy_only=False)).reshape(len(col), -1).copy())
    return tbl.column("text").to_pylist(), tbl.column("gold").to_pylist(), acts


def _cosine_lr(step, total, lr, warmup, min_ratio=0.1):
    import math
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a",
                "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"]


def nla_cfg():
    from huggingface_hub import snapshot_download

    from nla.config import load_nla_config
    tok = nla_tokenizer()
    d = snapshot_download(NLA36, allow_patterns=["av_sft_lora/*"])
    # Contract (marker token, templates, mse_scale) is model-agnostic for this
    # tokenizer; d_model/layer are identical for 3.8.
    return tok, load_nla_config(f"{d}/av_sft_lora", tok)


def actor_prompt_ids(tok, cfg):
    from nla.utils import build_prompt_text
    prompt = [{"role": "user", "content": cfg.actor_prompt_template.replace("{injection_char}", "<INJECT>")}]
    ids = tok.encode(build_prompt_text(prompt, cfg.injection_char, tok), add_special_tokens=False)
    assert ids.count(cfg.injection_token_id) == 1
    return ids


def _pad(seqs, pad, device=None, labels=None):
    import torch
    device = device or dev()
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        am[i, :len(s)] = 1
    return ids.to(device), am.to(device)


# ---------------------------------------------------------------------------
# Data-parallel plumbing: one process per GPU, grads all-reduced every step.
# ---------------------------------------------------------------------------

def _ddp_entry(rank, target, world, kwargs, port):
    import datetime

    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=90),
                                device_id=torch.device(f"cuda:{rank}"))
    try:
        target(rank, world, **kwargs)
    finally:
        if world > 1:
            dist.destroy_process_group()


def ddp_run(target, kwargs):
    """Run target(rank, world, **kwargs) on every visible GPU; commit the volume periodically."""
    import threading

    import torch
    import torch.multiprocessing as mp
    world = torch.cuda.device_count()
    print(f"ddp_run: {target.__name__} on {world} GPU(s)", flush=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(600):
            try:
                vol.commit()
            except Exception as e:  # never let a commit hiccup kill training
                print("commit failed:", e, flush=True)
    th = threading.Thread(target=committer, daemon=True)
    th.start()
    try:
        if world == 1:
            _ddp_entry(0, target, 1, kwargs, 29511)
        else:
            mp.spawn(_ddp_entry, args=(target, world, kwargs, 29511), nprocs=world, join=True)
    finally:
        stop.set()
        vol.commit()


def allreduce_grads(params, world):
    import torch
    import torch.distributed as dist
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if world == 1:
        return
    grads = [p.grad for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for g, f in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(f)


def broadcast_params(params, world):
    import torch.distributed as dist
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)


def allsum(vals, world):
    """Sum a list of floats across ranks."""
    import torch
    import torch.distributed as dist
    t = torch.tensor(vals, dtype=torch.float64, device=dev())
    if world > 1:
        dist.all_reduce(t)
    return t.tolist()


def barrier(world):
    import torch.distributed as dist
    if world > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# AV SFT
# ---------------------------------------------------------------------------

def _av_sft(rank, world, epochs=1.0, bs=64, lr=1e-4, lora_r=64, max_steps=0, run="av38_sft",
            eval_every=200, max_resp_tokens=320, resume=True, rewarm=20):
    """Inject act38 at the marker (Karvonen, layer-1 output); CE on the gold explanation.
    Global batch bs is split across ranks."""
    import json
    import math
    import random
    import time

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    from nla.utils import register_karvonen_hook

    torch.manual_seed(0)
    t0 = time.time()
    P = lambda *a: print(f"[r{rank}]", *a, flush=True) if rank == 0 else None
    tok, cfg = nla_cfg()
    _, G_tr, A_tr = _load_acts("train")
    _, G_va, A_va = _load_acts("val")
    P(f"train {len(G_tr)} val {len(G_va)} world={world}")
    base = load_text_lm(BASE38)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=16, lora_dropout=0.0, use_rslora=True,
                                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    save_dir = f"{CKPT}/{run}"
    start, prev_log = 0, None
    if resume and os.path.exists(f"{save_dir}/train_log.json") and os.path.exists(f"{save_dir}/adapter_model.safetensors"):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        prev_log = json.load(open(f"{save_dir}/train_log.json"))["log"]
        start = prev_log[-1]["step"]
        res = set_peft_model_state_dict(model, load_file(f"{save_dir}/adapter_model.safetensors"))
        assert not getattr(res, "unexpected_keys", []), "unexpected keys on resume"
        P(f"RESUMED LoRA from {save_dir} at step {start} (optimizer state re-initialized, {rewarm}-step re-warmup)")
    broadcast_params(params, world)
    P(f"trainable {sum(p.numel() for p in params)/1e6:.1f}M [{time.time()-t0:.0f}s]")
    vref = [None]
    register_karvonen_hook(model, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    pids = actor_prompt_ids(tok, cfg)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    resp_ids = lambda e: tok.encode(f"<explanation>\n{e}\n</explanation>" + tok.eos_token,
                                    add_special_tokens=False)[:max_resp_tokens]

    def loss_on(golds, acts, reduction="mean"):
        # vref stays set until the caller clears it AFTER backward: gradient
        # checkpointing re-runs layer 1 (and its injection hook) during backward.
        rs = [resp_ids(g) for g in golds]
        ids, am = _pad([pids + r for r in rs], pad)
        lab = torch.full_like(ids, -100)
        for i, r in enumerate(rs):
            lab[i, len(pids):len(pids) + len(r)] = torch.tensor(r, device=ids.device)
        vref[0] = acts.to(dev())
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits[:, :-1].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                               ignore_index=-100, reduction=reduction), (lab[:, 1:] != -100).sum()

    def val_loss(shuffle=False):
        model.eval()
        nv = min(512, len(G_va))
        perm = torch.roll(torch.arange(nv), 1)
        mine = list(range(rank, nv, world))
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(mine), 32):
                j = mine[s:s + 32]
                acts = A_va[perm[j]] if shuffle else A_va[j]
                l, nt = loss_on([G_va[i] for i in j], acts, reduction="sum")
                vref[0] = None
                tot += l.item()
                ntok += nt.item()
        model.train()
        a, b = allsum([tot, ntok], world)
        return a / b

    v0 = val_loss()
    P(f"step {start} val CE={v0:.4f} [{time.time()-t0:.0f}s]")
    log = prev_log or [{"step": 0, "val": v0}]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    n = len(G_tr)
    total = max_steps or int(math.ceil(n * epochs / bs))
    rng, order, ema = random.Random(0), [], None   # same seed on every rank -> same global order
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    for step in range(total):
        if len(order) < bs:
            o = list(range(n))
            rng.shuffle(o)
            order += o
        j, order = order[:bs], order[bs:]
        if step < start:          # replay the data order up to the resume point
            continue
        j = j[rank::world]
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, total, lr, 50) * (min(1.0, (step - start + 1) / rewarm) if start else 1.0)
        loss, _ = loss_on([G_tr[i] for i in j], A_tr[j])
        loss.backward()
        vref[0] = None
        allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = allsum([loss.item()], world)[0] / world
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % 20 == 0:
            P(f"step {step}/{total} loss={lv:.4f} ema={ema:.4f} gn={gn:.3f} "
              f"lr={opt.param_groups[0]['lr']:.2e} [{time.time()-t0:.0f}s]")
        if (step + 1) % eval_every == 0 or step + 1 == total:
            v, vs = val_loss(), val_loss(shuffle=True)
            P(f"step {step+1} val CE={v:.4f} shuffled-acts CE={vs:.4f} [{time.time()-t0:.0f}s]")
            log.append({"step": step + 1, "val": v, "val_shuffled": vs, "train_ema": ema})
            if rank == 0:
                model.save_pretrained(save_dir)
                json.dump({"log": log, "lora_r": lora_r, "lr": lr, "bs": bs, "base": BASE38, "world": world},
                          open(f"{save_dir}/train_log.json", "w"), indent=1)
            barrier(world)
    P(f"done [{time.time()-t0:.0f}s]")


@app.function(image=train_image, gpu=[G(4), G(2)], volumes={VOL: vol}, timeout=20 * 3600,
              secrets=secrets, memory=262144, cpu=32.0)
def train_av(**kw):
    ddp_run(_av_sft, kw)


# ---------------------------------------------------------------------------
# AR SFT (critic built from the 3.8 base)
# ---------------------------------------------------------------------------

def build_critic38(num_layers=LAYER + 1):
    """3.8 base truncated to blocks 0..42, final norm + lm_head stripped, identity value head."""
    import torch

    from nla.models import NLACriticModel
    base = load_text_lm(BASE38)
    base.model.layers = torch.nn.ModuleList(list(base.model.layers)[:num_layers])
    base.config.num_hidden_layers = num_layers
    base.config.layer_types = list(base.config.layer_types)[:num_layers]
    base.model.norm = torch.nn.Identity()
    base.lm_head = torch.nn.Identity()
    critic = NLACriticModel(base.config, base)
    with torch.no_grad():
        critic.value_head.weight.copy_(torch.eye(base.config.hidden_size))
    critic.value_head.float().to(dev())
    torch.cuda.empty_cache()
    return critic


def attach_ar_lora(critic, lora_r):
    from peft import LoraConfig, inject_adapter_in_model
    for p in critic.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=16, lora_dropout=0.0, use_rslora=True,
                                       target_modules=LORA_TARGETS), critic.backbone)
    critic.to(dev())


def ar_state(critic):
    return {k: v.detach().to("cpu") for k, v in critic.state_dict().items()
            if "lora_" in k or k.startswith("value_head.")}


def ar_params(critic):
    return [p for n_, p in critic.named_parameters() if "lora_" in n_ or n_.startswith("value_head.")]


def load_ar38(path, lora_r=64):
    from safetensors.torch import load_file
    critic = build_critic38()
    attach_ar_lora(critic, lora_r)
    sd = load_file(path)
    missing, unexpected = critic.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert not [k for k in missing if "lora_" in k or "value_head" in k], "AR lora/head not loaded"
    critic.eval()
    return critic


def ar_mse(critic, tok, cfg, texts, acts, sc):
    """Per-row normalized MSE of critic(explanation) vs acts. Caller handles grad mode."""
    from nla.schema import normalize_activation
    from nla.utils.critic import critic_predict
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    es = [tok.encode(cfg.critic_prompt_template.format(explanation=x), add_special_tokens=False)[-512:]
          for x in texts]
    ids, am = _pad(es, pad)
    pred = critic_predict(critic, ids, am, sc)
    return ((normalize_activation(pred, sc) - normalize_activation(acts.to(dev()).float(), sc)) ** 2).mean(-1)


def _ar_sft(rank, world, epochs=1.0, bs=64, lr=1e-4, head_lr=1e-4, lora_r=64, run="ar38_sft",
            eval_every=200, max_steps=0):
    import json
    import math
    import random
    import time

    import torch
    from safetensors.torch import save_file

    from nla.schema import compute_predict_mean_baselines, resolve_target_scale

    torch.manual_seed(0)
    t0 = time.time()
    P = lambda *a: print(f"[r{rank}]", *a, flush=True) if rank == 0 else None
    tok, cfg = nla_cfg()
    sc = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    _, G_tr, A_tr = _load_acts("train")
    _, G_va, A_va = _load_acts("val")
    _, base_va = compute_predict_mean_baselines(A_va, sc)
    critic = build_critic38()
    critic.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    critic.backbone.enable_input_require_grads()
    attach_ar_lora(critic, lora_r)
    params = ar_params(critic)
    for p in params:
        p.requires_grad_(True)
    broadcast_params(params, world)
    lora_p = [p for n_, p in critic.named_parameters() if "lora_" in n_]
    head_p = list(critic.value_head.parameters())
    P(f"trainable: lora {sum(p.numel() for p in lora_p)/1e6:.1f}M + head {sum(p.numel() for p in head_p)/1e6:.1f}M"
      f" | mem {torch.cuda.memory_allocated()/1e9:.1f}GB [{time.time()-t0:.0f}s]")
    opt = torch.optim.AdamW([{"params": lora_p, "base": lr}, {"params": head_p, "base": head_lr}],
                            lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

    def val_fve():
        critic.eval()
        mine = list(range(rank, len(G_va), world))
        with torch.no_grad():
            m = torch.cat([ar_mse(critic, tok, cfg, [G_va[i] for i in mine[s:s + 64]], A_va[mine[s:s + 64]], sc)
                           for s in range(0, len(mine), 64)])
        critic.train()
        a, b = allsum([m.sum().item(), len(m)], world)
        return (1 - (a / b) / base_va) * 100

    v0 = val_fve()
    P(f"step 0 val FVE={v0:.2f}% [{time.time()-t0:.0f}s]")
    log = [{"step": 0, "val_fve": v0}]
    n = len(G_tr)
    total = max_steps or int(math.ceil(n * epochs / bs))
    rng, order = random.Random(0), []
    save_dir = f"{CKPT}/{run}"
    os.makedirs(save_dir, exist_ok=True)
    critic.train()
    for step in range(total):
        if len(order) < bs:
            o = list(range(n))
            rng.shuffle(o)
            order += o
        j, order = order[:bs], order[bs:]
        j = j[rank::world]
        for gr in opt.param_groups:
            gr["lr"] = _cosine_lr(step, total, gr["base"], 50)
        loss = ar_mse(critic, tok, cfg, [G_tr[i] for i in j], A_tr[j], sc).mean()
        loss.backward()
        allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            lv = allsum([loss.item()], world)[0] / world
            P(f"step {step}/{total} mse={lv:.4f} fve~{(1-lv/base_va)*100:.1f}% gn={gn:.3f} [{time.time()-t0:.0f}s]")
        if (step + 1) % eval_every == 0 or step + 1 == total:
            v = val_fve()
            P(f"step {step+1} val FVE={v:.2f}% [{time.time()-t0:.0f}s]")
            log.append({"step": step + 1, "val_fve": v})
            if rank == 0:
                save_file(ar_state(critic), f"{save_dir}/ar_lora_value_head.safetensors")
                json.dump({"log": log, "lora_r": lora_r, "lr": lr, "head_lr": head_lr, "bs": bs,
                           "ar_num_layers": LAYER + 1, "final_norm_stripped": True, "base": BASE38,
                           "world": world}, open(f"{save_dir}/train_log.json", "w"), indent=1)
            barrier(world)
    P(f"done [{time.time()-t0:.0f}s]")


@app.function(image=train_image, gpu=G(4), volumes={VOL: vol}, timeout=12 * 3600, secrets=secrets,
              memory=196608, cpu=16.0)
def train_ar(**kw):
    ddp_run(_ar_sft, kw)


# ---------------------------------------------------------------------------
# GRPO — mirrors nla.train_rl_self_contained (reward = -MSE, -2 on failure,
# hinged length penalty, group-normalized advantages, on-policy surrogate +
# k3 KL to the SFT policy, one critic co-training step per batch).
# Data parallel: each rank owns batch_prompts/world prompt groups.
# ---------------------------------------------------------------------------

def load_av38(sft_run, merge=True):
    from peft import PeftModel
    base = load_text_lm(BASE38)
    m = PeftModel.from_pretrained(base, f"{CKPT}/{sft_run}")
    _check_lora_loaded(m)
    return m.merge_and_unload() if merge else m


def rollout(model, tok, cfg, pids, acts, max_new_tokens, temperature=1.0, bs=256, vref=None):
    import torch

    from nla.schema import EXPLANATION_RE
    stop = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.eval()
    outs = []
    for s in range(0, len(acts), bs):
        v = acts[s:s + bs]
        pt = torch.tensor([pids] * len(v), dtype=torch.long, device=dev())
        vref[0] = v.to(dev())
        try:
            with torch.no_grad():
                kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 \
                    else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
                o = model.generate(input_ids=pt, attention_mask=torch.ones_like(pt), max_new_tokens=max_new_tokens,
                                   eos_token_id=stop, pad_token_id=pad, use_cache=True, **kw)
        finally:
            vref[0] = None
        for r in range(len(v)):
            ids = o[r, len(pids):].tolist()
            cut = next((i for i, t in enumerate(ids) if t in stop), None)
            truncated = cut is None
            ids = ids if truncated else ids[:cut + 1]
            text = tok.decode(ids, skip_special_tokens=True)
            m = EXPLANATION_RE.search(text)
            outs.append({"ids": ids, "text": text, "expl": m.group(1).strip() if (m and not truncated) else None,
                         "truncated": truncated})
    return outs


def token_logps(model, pids, resp_list, acts, vref, pad, grad):
    """Per-token log-probs of each response. vref left SET (caller clears after backward)."""
    import torch
    ids, am = _pad([pids + r for r in resp_list], pad)
    vref[0] = acts.to(dev())
    with (torch.enable_grad() if grad else torch.no_grad()):
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
        P = len(pids)
        out = []
        for i, r in enumerate(resp_list):
            lg = logits[i, P - 1:P - 1 + len(r)].float()
            tgt = torch.tensor(r, device=lg.device)
            out.append(torch.log_softmax(lg, -1).gather(-1, tgt[:, None])[:, 0])
    return out


def _rl(rank, world, av_sft="av38_sft", ar_sft="ar38_sft", run="rl38", num_steps=200, prompts_per_rank=16,
        group_size=8, max_new_tokens=256, lr=1e-4, critic_lr=8e-5, kl_beta=0.01, length_penalty=0.01,
        lora_r=64, mb=16, critic_mb=32, eval_every=10, eval_n=256, save_every=10, max_hours=10.0,
        max_usd=1e9, usd_per_gpu_hour=6.25, resume=True):
    import json
    import math
    import random
    import time

    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file

    from nla.schema import compute_predict_mean_baselines, resolve_target_scale
    from nla.utils import register_karvonen_hook

    batch_prompts = prompts_per_rank * world
    torch.manual_seed(0)
    t0 = time.time()
    P = lambda *a: print(f"[r{rank}]", *a, flush=True) if rank == 0 else None
    P(f"GRPO: {world} GPUs x {prompts_per_rank} prompts x {group_size} = {batch_prompts * group_size} rollouts/step; "
      f"stop at {num_steps} steps / {max_hours}h / ${max_usd:.0f}")
    tok, cfg = nla_cfg()
    sc = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    _, _, A_rl = _load_acts("rl")
    _, _, A_va = _load_acts("val")
    _, base_rl = compute_predict_mean_baselines(A_rl[:20000], sc)
    _, base_ev = compute_predict_mean_baselines(A_va, sc)
    ev_mine = list(range(rank, min(eval_n, len(A_va)), world))
    save_dir = f"{CKPT}/{run}"
    os.makedirs(save_dir, exist_ok=True)
    state_path = f"{save_dir}/state.json"
    start = json.load(open(state_path))["step"] if (resume and os.path.exists(state_path)) else 0

    # actor = merged SFT AV + RL LoRA; reference = adapter disabled (= the SFT AV)
    base = load_av38(av_sft, merge=True)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    if start:
        actor = PeftModel.from_pretrained(base, save_dir, is_trainable=True)
    else:
        actor = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=16, lora_dropout=0.0, use_rslora=True,
                                                target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    pids = actor_prompt_ids(tok, cfg)
    a_params = [p for p in actor.parameters() if p.requires_grad]
    broadcast_params(a_params, world)
    opt = torch.optim.AdamW(a_params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

    critic = load_ar38(f"{save_dir}/critic.safetensors" if start else f"{CKPT}/{ar_sft}/ar_lora_value_head.safetensors")
    critic.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    critic.backbone.enable_input_require_grads()
    c_params = ar_params(critic)
    for p in c_params:
        p.requires_grad_(True)
    copt = torch.optim.AdamW(c_params, lr=critic_lr, betas=(0.9, 0.99), weight_decay=0.0)
    # ranks hold identical optimizer state (all-reduced grads, synced weights): fall back to rank 0's
    opath = next((p_ for p_ in (f"{save_dir}/optim_r{rank}.pt", f"{save_dir}/optim_r0.pt") if os.path.exists(p_)), None)
    if start and opath:
        o = torch.load(opath, map_location=dev())
        opt.load_state_dict(o["actor"])
        copt.load_state_dict(o["critic"])
    P(f"setup done (start={start}) [{time.time()-t0:.0f}s] mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    def score(expls, acts):
        critic.eval()
        out = [None] * len(expls)
        idx = [i for i, e in enumerate(expls) if e]
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                j = idx[s:s + 64]
                m = ar_mse(critic, tok, cfg, [expls[i] for i in j], acts[j], sc)
                for k, i in enumerate(j):
                    out[i] = m[k].item()
        return out

    def run_eval(step):
        outs = rollout(actor, tok, cfg, pids, A_va[ev_mine], max_new_tokens, temperature=0.0, vref=vref)
        m = score([o["expl"] for o in outs], A_va[ev_mine])
        v = [x for x in m if x is not None]
        s_m, n_v, n_t, s_len = allsum([sum(v), len(v), len(m), sum(len(o["ids"]) for o in outs)], world)
        return {"step": step, "eval_fve": (1 - (s_m / max(n_v, 1)) / base_ev) * 100, "eval_extract": n_v / n_t,
                "eval_len": s_len / n_t, "sample": outs[0]["text"][:600]}

    log = json.load(open(f"{save_dir}/log.json")) if (start and os.path.exists(f"{save_dir}/log.json")) else []
    if start == 0:
        e = run_eval(0)
        P(f"[eval] step 0 FVE={e['eval_fve']:.2f}% extract={e['eval_extract']:.2f} len={e['eval_len']:.0f} "
          f"[{time.time()-t0:.0f}s]\n  sample: {e['sample']!r}")
        log.append(e)
    thr = max_new_tokens - 64
    pr = batch_prompts // world
    for step in range(start, num_steps):
        ts = time.time()
        hrs = (ts - t0) / 3600
        stop_now = allsum([float(hrs > max_hours or hrs * world * usd_per_gpu_hour > max_usd)], world)[0] > 0
        if stop_now:
            P(f"time/budget limit reached at step {step} ({hrs:.2f}h, ~${hrs * world * usd_per_gpu_hour:.0f})")
            if step > start and step % save_every != 0:
                torch.save({"actor": opt.state_dict(), "critic": copt.state_dict()}, f"{save_dir}/optim_r{rank}.pt")
                if rank == 0:
                    actor.save_pretrained(save_dir)
                    save_file(ar_state(critic), f"{save_dir}/critic.safetensors")
                    json.dump({"step": step, "av_sft": av_sft, "ar_sft": ar_sft, "lora_r": lora_r, "world": world},
                              open(state_path, "w"))
                    json.dump(log, open(f"{save_dir}/log.json", "w"), indent=1)
                barrier(world)
            break
        rows = random.Random(1000003 * step + rank).sample(range(len(A_rl)), pr)
        acts = A_rl[rows].repeat_interleave(group_size, 0)
        outs = rollout(actor, tok, cfg, pids, acts, max_new_tokens, temperature=1.0, vref=vref)
        t_roll = time.time() - ts
        mses = score([o["expl"] for o in outs], acts)
        rew = torch.tensor([-2.0 if m is None else -m for m in mses])
        lens = torch.tensor([float(len(o["ids"])) for o in outs])
        shaped = rew - length_penalty * (lens - thr).clamp_min(0)
        adv = torch.zeros_like(shaped)
        for g in range(pr):
            sl = slice(g * group_size, (g + 1) * group_size)
            r = shaped[sl]
            adv[sl] = (r - r.mean()) / (r.std() + 1e-6)
        # ---- actor update ----
        actor.train()
        keep = [i for i, o in enumerate(outs) if len(o["ids"]) > 0]
        kls = []
        for s in range(0, len(keep), mb):
            j = keep[s:s + mb]
            resp = [outs[i]["ids"] for i in j]
            with actor.disable_adapter():
                ref = token_logps(actor, pids, resp, acts[j], vref, pad, grad=False)
            vref[0] = None
            new = token_logps(actor, pids, resp, acts[j], vref, pad, grad=True)
            loss = 0.0
            for k, i in enumerate(j):
                d = (ref[k] - new[k]).clamp(max=12.0)
                kl = torch.exp(d) - d - 1.0
                loss = loss + (-(adv[i].item() * new[k] - kl_beta * kl)).mean()
                kls.append(kl.detach().mean().item())
            (loss / max(1, len(keep))).backward()
            vref[0] = None
        allreduce_grads(a_params, world)
        gn = torch.nn.utils.clip_grad_norm_(a_params, 1.0)
        if math.isfinite(gn.item()):
            opt.step()
        opt.zero_grad(set_to_none=True)
        # ---- critic co-training: one step on this batch's explanations ----
        critic.train()
        valid = [i for i, m in enumerate(mses) if m is not None]
        closs = 0.0
        for s in range(0, len(valid), critic_mb):
            j = valid[s:s + critic_mb]
            l = ar_mse(critic, tok, cfg, [outs[i]["expl"] for i in j], acts[j], sc).sum() / max(1, len(valid))
            l.backward()
            closs += l.item()
        allreduce_grads(c_params, world)
        cgn = torch.nn.utils.clip_grad_norm_(c_params, 1.0)
        if math.isfinite(cgn.item()):
            copt.step()
        copt.zero_grad(set_to_none=True)
        vm = [m for m in mses if m is not None]
        S = allsum([sum(vm), len(vm), len(mses), sum(o["truncated"] for o in outs), lens.sum().item(),
                    sum(kls), len(kls), closs], world)
        rec = {"step": step + 1, "fve": (1 - (S[0] / max(S[1], 1)) / base_rl) * 100, "extract": S[1] / S[2],
               "trunc": S[3] / S[2], "len": S[4] / S[2], "kl": S[5] / max(S[6], 1), "gn": gn.item(),
               "critic_mse": S[7] / world, "t_roll": t_roll, "t_step": time.time() - ts}
        P(f"step {step+1} fve={rec['fve']:.2f}% ext={rec['extract']:.2f} trunc={rec['trunc']:.2f} "
          f"len={rec['len']:.0f} kl={rec['kl']:.4f} gn={rec['gn']:.3f} cmse={rec['critic_mse']:.4f} "
          f"roll={t_roll:.0f}s step={rec['t_step']:.0f}s [{(time.time()-t0)/60:.1f}m]")
        log.append(rec)
        if (step + 1) % eval_every == 0 or step + 1 == num_steps:
            e = run_eval(step + 1)
            P(f"[eval] step {step+1} FVE={e['eval_fve']:.2f}% extract={e['eval_extract']:.2f} "
              f"len={e['eval_len']:.0f}\n  sample: {e['sample']!r}")
            log.append(e)
        if (step + 1) % save_every == 0 or step + 1 == num_steps:
            torch.save({"actor": opt.state_dict(), "critic": copt.state_dict()}, f"{save_dir}/optim_r{rank}.pt")
            if rank == 0:
                actor.save_pretrained(save_dir)
                save_file(ar_state(critic), f"{save_dir}/critic.safetensors")
                json.dump({"step": step + 1, "av_sft": av_sft, "ar_sft": ar_sft, "lora_r": lora_r, "world": world},
                          open(state_path, "w"))
                json.dump(log, open(f"{save_dir}/log.json", "w"), indent=1)
            barrier(world)
    P(f"done [{(time.time()-t0)/60:.1f}m]")


@app.function(image=hopper_image, gpu=(["B200:8", "H200:8", "B200:4", "H200:4"] if GPU == "B200" else [G(8), G(4)]), volumes={VOL: vol},
              timeout=24 * 3600, secrets=secrets, memory=262144, cpu=32.0)
def train_rl(**kw):
    # hopper_image = train_image + triton 3.7.1: fla's gated-delta backward is correct on H200 with it
    # (check_hopper_triton: fwd/bwd within 0.6% of a naive fp32 recurrence); B200 unaffected.
    ddp_run(_rl, kw)


@app.function(image=train_image, gpu=G(), volumes={VOL: vol}, timeout=3 * 3600, secrets=secrets)
def evaluate(av_sft: str = "av38_sft", rl_run: str = "", critic: str = "", n: int = 256,
             max_new_tokens: int = 256, tag: str = "eval"):
    """Held-out FVE on the 256 rl-split docs from the transfer test (doc bucket < 5,
    never trained on), greedy. Shuffled control: AV reads another row's activation,
    scored against the true one (should collapse if the AV really reads the vector)."""
    import json

    import numpy as np
    import torch
    from peft import PeftModel

    from nla.schema import compute_predict_mean_baselines, resolve_target_scale
    from nla.utils import register_karvonen_hook

    tok, cfg = nla_cfg()
    sc = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    rows = heldout_rows(n)
    model = load_text_lm(BASE38)
    a = extract_last_token(model, tok, [r["text"] for r in rows])   # targets from the PRISTINE base
    model = PeftModel.from_pretrained(model, f"{CKPT}/{av_sft}")
    _check_lora_loaded(model)
    model = model.merge_and_unload()
    if rl_run:
        model = PeftModel.from_pretrained(model, f"{CKPT}/{rl_run}")
        _check_lora_loaded(model)
    model.eval()
    vref = [None]
    register_karvonen_hook(model, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    pids = actor_prompt_ids(tok, cfg)
    outs = rollout(model, tok, cfg, pids, a, max_new_tokens, temperature=0.0, vref=vref)
    outs_sh = rollout(model, tok, cfg, pids, a[torch.roll(torch.arange(len(a)), 1)], max_new_tokens,
                      temperature=0.0, vref=vref)
    del model
    torch.cuda.empty_cache()
    cpath = critic or (f"{CKPT}/{rl_run}/critic.safetensors" if rl_run
                       else f"{CKPT}/ar38_sft/ar_lora_value_head.safetensors")
    cr = load_ar38(cpath)
    _, b = compute_predict_mean_baselines(a, sc)

    def f(expls):
        idx = [i for i, e in enumerate(expls) if e]
        m = []
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                j = idx[s:s + 64]
                m += ar_mse(cr, tok, cfg, [expls[i] for i in j], a[j], sc).tolist()
        return ((1 - np.mean(m) / b) * 100 if m else float("nan")), len(m) / len(expls)

    res = {}
    res["fve"], res["extract"] = f([o["expl"] for o in outs])
    res["shuffled_fve"], res["shuffled_extract"] = f([o["expl"] for o in outs_sh])
    res["mean_len"] = float(np.mean([len(o["ids"]) for o in outs]))
    print(json.dumps(res, indent=1), flush=True)
    os.makedirs(f"{VOL}/results", exist_ok=True)
    json.dump({"summary": res, "av_sft": av_sft, "rl_run": rl_run, "critic": cpath,
               "rows": [{"text": r["text"][-600:], "explanation": outs[i]["expl"], "raw": outs[i]["text"]}
                        for i, r in enumerate(rows)]}, open(f"{VOL}/results/{tag}.json", "w"), indent=1)
    vol.commit()
    for i in range(3):
        print(f"\n--- …{rows[i]['text'][-200:]!r}\n>>> {outs[i]['text'][:600]!r}", flush=True)
    return res


@app.local_entrypoint()
def launch(stage: str, gpus: str = "default", args: str = "{}"):
    """modal run --detach modal_nla38.py::launch --stage av --gpus B200:4,H200:4 --args '{"max_steps": 20}'
    gpus: comma-separated fallback list (first available wins)."""
    import json
    fn = {"av": train_av, "ar": train_ar, "rl": train_rl}[stage]
    if gpus == "default":   # use the decorator's GPU spec (may be a fallback list)
        fn.remote(**json.loads(args))
    else:
        fn.with_options(gpu=gpus if ":" in gpus else f"B200:{gpus}").remote(**json.loads(args))


pub_image = image.add_local_file(HERE / "nla_qwen38.py", "/root/nla_qwen38.py").add_local_dir(
    HERE, "/root/pub", ignore=lambda p: not str(p).endswith("README_HF.md"))


@app.function(image=pub_image, volumes={VOL: vol}, timeout=3600, cpu=8.0, memory=32768, secrets=secrets)
def export(av_sft: str = "av38_sft", ar_sft: str = "ar38_sft", rl_run: str = "rl38", out: str = "export"):
    """Bundle checkpoints + sidecar in a self-describing layout under /vol/<out>."""
    import json
    import shutil

    import yaml
    from huggingface_hub import snapshot_download

    dst = f"{VOL}/{out}"
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    shutil.copytree(f"{CKPT}/{av_sft}", f"{dst}/av_sft_lora")
    shutil.copytree(f"{CKPT}/{ar_sft}", f"{dst}/ar_sft_critic")
    if rl_run and os.path.exists(f"{CKPT}/{rl_run}/adapter_model.safetensors"):
        os.makedirs(f"{dst}/av_rl_lora")
        for f in ("adapter_model.safetensors", "adapter_config.json", "log.json", "state.json"):
            if os.path.exists(f"{CKPT}/{rl_run}/{f}"):
                shutil.copy(f"{CKPT}/{rl_run}/{f}", f"{dst}/av_rl_lora/{f}")
        os.makedirs(f"{dst}/rl_critic")
        shutil.copy(f"{CKPT}/{rl_run}/critic.safetensors", f"{dst}/rl_critic/ar_lora_value_head.safetensors")
    d = snapshot_download(NLA36, allow_patterns=["nla_meta.yaml"])
    meta = yaml.safe_load(open(f"{d}/nla_meta.yaml"))
    meta["dataset_id"] = f"Qwen3.8-27B_L{LAYER}_finefineweb_sonnet46_gold"
    meta["extraction"]["base_model"] = BASE38
    meta.pop("parent_datasets", None)
    meta["created_by"] = "modal_nla38.py"
    yaml.safe_dump(meta, open(f"{dst}/nla_meta.yaml", "w"), sort_keys=False, allow_unicode=True)
    # self-contained config for nla_qwen38.py (no EasyNLA / sidecar parsing needed)
    import sys
    sys.path.insert(0, "/root/EasyNLA")
    tok, cfg = nla_cfg()
    rendered = tok.decode(actor_prompt_ids(tok, cfg))
    ids = actor_prompt_ids(tok, cfg)
    assert tok.encode(rendered, add_special_tokens=False) == ids
    json.dump({"base_model": BASE38, "layer": LAYER, "d_model": cfg.d_model,
               "injection_char": cfg.injection_char, "injection_token_id": cfg.injection_token_id,
               "injection": "additive norm-matched (Karvonen) at block-1 output",
               "actor_prompt_rendered": rendered, "actor_prompt_ids": ids,
               "critic_prompt_template": cfg.critic_prompt_template,
               "ar_lora_r": json.load(open(f"{CKPT}/{ar_sft}/train_log.json"))["lora_r"],
               "ar_num_layers": LAYER + 1, "lora_targets": LORA_TARGETS},
              open(f"{dst}/nla_config.json", "w"), indent=1, ensure_ascii=False)
    shutil.copy("/root/nla_qwen38.py", f"{dst}/nla_qwen38.py")
    if os.path.exists("/root/pub/README_HF.md"):
        shutil.copy("/root/pub/README_HF.md", f"{dst}/README.md")
    for r in ("eval_sft", "eval_rl"):
        if os.path.exists(f"{VOL}/results/{r}.json"):
            shutil.copy(f"{VOL}/results/{r}.json", f"{dst}/{r}.json")
    vol.commit()
    for root, _, files in os.walk(dst):
        for f in files:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p[len(dst)+1:]}")


@app.function(image=pub_image, gpu=G(),
              volumes={VOL: vol}, timeout=3600, secrets=secrets)
def test_infer(src: str = "export", stage: str = "sft", n: int = 64):
    """Run the standalone nla_qwen38.py against an exported bundle on held-out docs."""
    import sys
    import time
    sys.path.insert(0, "/root")
    import torch
    from nla_qwen38 import NLA
    t0 = time.time()
    nla = NLA(f"{VOL}/{src}", stage=stage)
    print(f"loaded stage={nla.stage} [{time.time()-t0:.0f}s] mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    rows = heldout_rows(n)
    acts = nla.extract([r["text"] for r in rows])
    expl = nla.explain(acts)
    rec = nla.reconstruct(expl)
    print(f"standalone FVE={nla.fve(rec, acts)*100:.2f}% extract={sum(e is not None for e in expl)/n:.2f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    for i in range(4):
        print(f"\n--- ...{rows[i]['text'][-160:]!r}\n>>> {expl[i]!r}", flush=True)


@app.function(image=pub_image, volumes={VOL: vol}, timeout=2 * 3600, cpu=8.0, memory=32768, secrets=secrets)
def push_hf(src: str = "export", repo: str = "gereon/qwen3.8-27b-nla-L42", private: bool = True,
            message: str = "update"):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(folder_path=f"{VOL}/{src}", repo_id=repo, commit_message=message)
    print("pushed", repo, "private" if private else "public", flush=True)


@app.function(volumes={VOL: vol}, timeout=2 * 3600, cpu=8.0, memory=16384, secrets=secrets)
def push_data(repo: str = "gereon/qwen3.8-27b-nla-L42-data", private: bool = True):
    """Back up the extracted activations, row pools and result JSONs to a private HF dataset."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=DATA, path_in_repo="data", repo_id=repo, repo_type="dataset",
                      commit_message="Qwen3.8-27B L42 activations + pools")
    api.upload_folder(folder_path=f"{VOL}/results", path_in_repo="results", repo_id=repo, repo_type="dataset",
                      allow_patterns=["*.json"], commit_message="eval results")
    print("pushed", repo, flush=True)



@app.function(image=hopper_image, gpu="H200", timeout=1200)
def check_hopper_triton():
    """fla gated-delta fwd+bwd (GVA, bf16) on Hopper with triton 3.7.1 vs a naive fp32 recurrence."""
    import torch
    import torch.nn.functional as F
    import triton
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    print("torch", torch.__version__, "triton", triton.__version__, torch.cuda.get_device_name(), flush=True)

    def naive(q, k, v, g, beta):
        # q,k [B,T,HK,D]; v [B,T,HV,Dv]; g,beta [B,T,HV]. GVA: v-head h reads qk-head h // (HV/HK)
        B, T, HK, D = q.shape
        HV, Dv = v.shape[2], v.shape[3]
        r = HV // HK
        q = F.normalize(q.float(), dim=-1).repeat_interleave(r, 2) * D ** -0.5
        k = F.normalize(k.float(), dim=-1).repeat_interleave(r, 2)
        v, g, beta = v.float(), g.float(), beta.float()
        S = torch.zeros(B, HV, D, Dv, device=q.device)
        out = []
        for t in range(T):
            S = S * g[:, t].exp()[..., None, None]
            kt, vt = k[:, t], v[:, t]
            vn = beta[:, t][..., None] * (vt - torch.einsum("bhd,bhdv->bhv", kt, S))
            S = S + torch.einsum("bhd,bhv->bhdv", kt, vn)
            out.append(torch.einsum("bhd,bhdv->bhv", q[:, t], S))
        return torch.stack(out, 1)

    torch.manual_seed(0)
    B, T, HK, HV, D = 2, 200, 4, 12, 128
    q = torch.randn(B, T, HK, D, device="cuda")
    k = torch.randn(B, T, HK, D, device="cuda")
    v = torch.randn(B, T, HV, D, device="cuda")
    g = -F.softplus(torch.randn(B, T, HV, device="cuda")) * 0.1
    beta = torch.rand(B, T, HV, device="cuda")
    w = torch.randn(B, T, HV, D, device="cuda")
    res = {}
    for name in ("fla", "naive"):
        qq, kk, vv = (x.clone().to(torch.bfloat16 if name == "fla" else torch.float32).requires_grad_(True)
                      for x in (q, k, v))
        if name == "fla":
            o, _ = chunk_gated_delta_rule(qq, kk, vv, g=g, beta=beta.to(torch.bfloat16), use_qk_l2norm_in_kernel=True)
        else:
            o = naive(qq, kk, vv, g, beta)
        (o.float() * w).sum().backward()
        res[name] = (o.float(), qq.grad.float(), kk.grad.float(), vv.grad.float())
    worst = 0.0
    for j, n in enumerate(("out", "dq", "dk", "dv")):
        a, b = res["fla"][j], res["naive"][j]
        e = ((a - b).norm() / b.norm()).item()
        worst = max(worst, e)
        print(n, "rel_err", e, flush=True)
    print("VERDICT", "OK" if worst < 0.05 else "MISMATCH", flush=True)


@app.function(volumes={VOL: vol}, timeout=2 * 3600, cpu=8.0, memory=16384, secrets=secrets)
def push_ckpts(repo: str = "gereon/qwen3.8-27b-nla-L42-data", run: str = "rl38"):
    """Back up resumable training state (SFT LoRAs/critic + RL LoRA/critic/optimizer rank 0) to HF."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for d in ("av38_sft", "ar38_sft"):
        api.upload_folder(folder_path=f"{CKPT}/{d}", path_in_repo=f"ckpts/{d}", repo_id=repo, repo_type="dataset",
                          commit_message=f"ckpt {d}")
    api.upload_folder(folder_path=f"{CKPT}/{run}", path_in_repo=f"ckpts/{run}", repo_id=repo, repo_type="dataset",
                      ignore_patterns=["optim_r[1-9]*.pt"], commit_message=f"resumable {run} checkpoint")
    print("pushed ckpts to", repo, flush=True)


@app.function(volumes={VOL: vol}, timeout=3 * 3600, cpu=8.0, memory=32768, secrets=secrets)
def restore(repo: str = "gereon/qwen3.8-27b-nla-L42-data"):
    """Rebuild this pipeline's volume in a NEW Modal workspace, entirely from Hugging Face."""
    import shutil

    from huggingface_hub import snapshot_download
    print(snapshot_download(BASE38, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]), flush=True)
    vol.commit()
    snapshot_download(NLA36, allow_patterns=["nla_meta.yaml", "av_sft_lora/*"])
    snapshot_download(DATA8B, repo_type="dataset", allow_patterns=["rl_shuf*"])
    d = snapshot_download(repo, repo_type="dataset")
    for sub, dst in (("data", DATA), ("ckpts", CKPT), ("results", f"{VOL}/results")):
        if os.path.isdir(f"{d}/{sub}"):
            shutil.copytree(f"{d}/{sub}", dst, dirs_exist_ok=True)
    vol.commit()
    for root, _, files in os.walk(CKPT):
        for f in files:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p}", flush=True)
