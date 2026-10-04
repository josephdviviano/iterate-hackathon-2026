"""Trainer-side model utilities for RLTL;DR policy updates (on the trainer GPU).

The policy is Qwen3.8-27B served by vLLM as block-FP8 (Qwen3_5ForConditionalGeneration, TP=2). The trainer keeps
a frozen bf16 copy of the *dequantized* FP8 checkpoint (so both sides use near-identical base weights) and trains a
LoRA adapter on the text model only. Each new policy version is exported as a PEFT adapter and hot-loaded into vLLM
under a new name.

Public API
----------
load_policy(base_dir, ...)                    -> (peft_model, tokenizer)
token_logprobs(peft_model, ids, positions)    -> log p(ids[t] | ids[:t]) for t in positions (differentiable)
export_adapter(peft_model, out_dir)           -> vLLM-loadable PEFT dir (bf16, multimodal key names), atomic
load_adapter_weights(peft_model, adapter_dir) -> inverse of export_adapter (optionally into a second adapter slot)
adapter_context(peft_model, name)             -> temporarily compute with another adapter (e.g. an old policy)
save_trainer_state / load_trainer_state       -> LoRA fp32 weights + optimizer state + metadata (crash recovery)

Numerics / memory design (measured numbers are in README_trainer_utils.md):
* LoRA master weights are fp32 (PEFT autocast_adapter_dtype). By default the LoRA matmuls run in bf16 under
  torch.autocast with PEFT's input up-cast disabled: this avoids saving an fp32 copy of every LoRA input for
  backward (large at long context) and matches vLLM, which applies LoRA in bf16. Everything else in the network
  is already bf16/fp32-explicit, so autocast does not change the base model's numerics (verified: identical
  base log-probs with and without autocast).
* Log-probs are computed from the final hidden states with a custom autograd function that walks the vocabulary
  projection in chunks of positions, so the full [L, 248320] logits tensor never exists. Logits are produced by a
  bf16 GEMM with fp32 output (no bf16 rounding of the logits); the backward recomputes each chunk's logits in place.
* Gradient checkpointing (non-reentrant) per decoder layer; optional TRL activation offload of the per-layer
  checkpoint inputs (10 KiB/token/layer) to pinned CPU memory for long sequences.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",          # gated full attention (16 layers)
    "gate_proj", "up_proj", "down_proj",             # MLP (64 layers)
    "in_proj_qkv", "in_proj_z", "out_proj",          # Gated DeltaNet (48 layers)
)
# LoRA parameter count per unit rank for DEFAULT_TARGET_MODULES on Qwen3.8-27B (sum of in+out over all targets).
LORA_PARAMS_PER_RANK = 6_799_360

# Checkpoint (multimodal) naming vs. the text-only CausalLM naming used in the trainer.
CKPT_TEXT_PREFIX = "model.language_model."
TRAINER_LAYER_PREFIX = "base_model.model.model.layers."
# vLLM serves Qwen3_5ForConditionalGeneration: it strips "base_model.model." and maps "model.language_model." ->
# "language_model.model.". Keys without "language_model" match no module and the adapter silently does nothing.
VLLM_LAYER_PREFIX = "base_model.model.model.language_model.layers."

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

_ORIG_CLASSES: dict[str, Any] = {}


# ----------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------
def _fla_implementation_module() -> str:
    """Module that transformers' Qwen3.5 DeltaNet chunk kernel resolved to (must be FLA, not the torch fallback)."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    impl = inspect.getclosurevars(mq.torch_chunk_gated_delta_rule).nonlocals.get("implementation")
    return getattr(impl, "__module__", "") or ""


def _set_liger(enabled: bool) -> None:
    """Patch (or restore) the Qwen3.5 RMSNorm and SwiGLU MLP classes. Must run before the model is constructed.

    Liger's RMSNorm (offset 1.0, gemma casting) keeps only x and rstd for backward instead of several fp32 copies,
    and its SiLU*mul kernel avoids saving silu(gate). RoPE and fused CE are not used (RoPE is unsupported for
    Qwen3.5 in Liger 0.8.4; the chunked log-prob function below replaces the loss).
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if not _ORIG_CLASSES:
        _ORIG_CLASSES.update(rms=mq.Qwen3_5RMSNorm, mlp=mq.Qwen3_5MLP)
    if enabled:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5(rope=False, rms_norm=True, swiglu=True,
                                      cross_entropy=False, fused_linear_cross_entropy=False)
    else:
        mq.Qwen3_5RMSNorm, mq.Qwen3_5MLP = _ORIG_CLASSES["rms"], _ORIG_CLASSES["mlp"]


def _resolve_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _expected_text_keys(base_dir: str) -> tuple[set[str], dict[str, str], int, int]:
    """Map checkpoint keys to Qwen3_5ForCausalLM state-dict names. Returns (expected names, name->ckpt key map,
    #visual keys skipped, #mtp keys skipped)."""
    weight_map = json.loads((Path(base_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    name_to_ckpt, n_visual, n_mtp = {}, 0, 0
    for k in weight_map:
        if k.startswith(CKPT_TEXT_PREFIX):
            name_to_ckpt["model." + k[len(CKPT_TEXT_PREFIX):]] = k
        elif k == "lm_head.weight":
            name_to_ckpt[k] = k
        elif k.startswith("model.visual."):
            n_visual += 1
        elif k.startswith("mtp."):
            n_mtp += 1
        else:
            raise AssertionError(f"unrecognised checkpoint key {k!r}")
    return set(name_to_ckpt), name_to_ckpt, n_visual, n_mtp


@torch.no_grad()
def verify_base_weights(model: torch.nn.Module, base_dir: str, names: Iterable[str] | None = None) -> int:
    """Check that the loaded CausalLM parameters equal the checkpoint tensors bit-for-bit.

    `model` is the bare Qwen3_5ForCausalLM (or the PEFT wrapper: LoRA-wrapped linears expose `.base_layer`).
    `names=None` checks every tensor (~54 GB read; ~1 min from page cache). Returns the number of tensors checked.
    """
    weight_map = json.loads((Path(base_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    _, name_to_ckpt, _, _ = _expected_text_keys(base_dir)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = {}
    for n, p in inner.named_parameters():
        params[n.replace(".base_layer.", ".")] = p
    names = sorted(name_to_ckpt) if names is None else list(names)
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(weight_map[name_to_ckpt[n]], []).append(n)
    for shard, shard_names in by_shard.items():
        with safe_open(str(Path(base_dir) / shard), "pt", device=str(next(iter(params.values())).device)) as f:
            for n in shard_names:
                ref = f.get_tensor(name_to_ckpt[n])
                p = params[n]
                if p.shape != ref.shape or p.dtype != ref.dtype or not torch.equal(p, ref):
                    raise AssertionError(f"weight mismatch for {n} ({p.dtype}{tuple(p.shape)} vs "
                                         f"{ref.dtype}{tuple(ref.shape)})")
    return len(names)


def load_policy(
    base_dir: str,
    lora_r: int = 32,
    lora_alpha: int = 32,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    device: str | torch.device = "cuda",
    gradient_checkpointing: bool = True,
    use_liger: bool = True,
    lora_compute_dtype: torch.dtype = torch.bfloat16,
    verify_weights: str = "spot",
    log: bool = True,
):
    """Load the frozen bf16 text model + a fresh LoRA adapter ("default", B=0, i.e. identical to the base).

    Args:
        base_dir: dequantized checkpoint dir (multimodal config with `text_config`, keys `model.language_model.*`).
        lora_compute_dtype: torch.bfloat16 (default: LoRA matmuls in bf16 under autocast, fp32 master weights) or
            torch.float32 (PEFT default: LoRA inputs up-cast to fp32; more activation memory).
        verify_weights: "spot" (compare a handful of tensors with the checkpoint), "full" (all), or "none".
    Returns:
        (peft_model, tokenizer). The model is in train() mode; only LoRA params require grad (fp32).
    """
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if lora_compute_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("lora_compute_dtype must be torch.bfloat16 or torch.float32")
    t0 = time.time()
    device = _resolve_device(device)

    fla_mod = _fla_implementation_module()
    assert fla_mod.startswith("fla."), (
        f"transformers' chunk_gated_delta_rule resolved to {fla_mod!r}, not FLA (is flash-linear-attention "
        "installed?). The torch fallback is >10x slower.")
    _set_liger(use_liger)

    cfg = AutoConfig.from_pretrained(base_dir)
    text_cfg = getattr(cfg, "text_config", cfg)
    assert text_cfg.model_type == "qwen3_5_text", text_cfg.model_type
    assert not getattr(text_cfg, "tie_word_embeddings", False)
    text_cfg.use_cache = False

    model, info = mq.Qwen3_5ForCausalLM.from_pretrained(
        base_dir, config=text_cfg, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="sdpa", output_loading_info=True)
    t_load = time.time() - t0

    # --- key verification: transformers' report + an independent check against the checkpoint index ---------
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith(("model.visual.", "mtp.", "visual."))]
    assert not info.get("missing_keys"), f"missing keys: {sorted(info['missing_keys'])[:20]}"
    assert not unexpected, f"unexpected keys: {unexpected[:20]}"
    assert not info.get("mismatched_keys"), f"mismatched keys: {info['mismatched_keys'][:20]}"
    assert not info.get("error_msgs"), info["error_msgs"]
    expected, _, n_visual, n_mtp = _expected_text_keys(base_dir)
    have = set(model.state_dict().keys())
    assert have == expected, (f"state-dict/checkpoint key mismatch: only in model {sorted(have - expected)[:10]}, "
                              f"only in ckpt {sorted(expected - have)[:10]}")
    n_layers = text_cfg.num_hidden_layers
    if verify_weights == "spot":
        spot = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                "model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.0.linear_attn.A_log",
                "model.layers.3.self_attn.q_proj.weight", f"model.layers.{n_layers - 1}.mlp.down_proj.weight",
                f"model.layers.{n_layers - 1}.self_attn.o_proj.weight"]
        n_verified = verify_base_weights(model, base_dir, spot)
    elif verify_weights == "full":
        n_verified = verify_base_weights(model, base_dir)
    elif verify_weights == "none":
        n_verified = 0
    else:
        raise ValueError(f"verify_weights={verify_weights!r}")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=list(target_modules), task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lcfg)  # autocast_adapter_dtype=True -> fp32 LoRA weights
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable and all(("lora_A" in n or "lora_B" in n) and p.dtype == torch.float32 for n, p in trainable)
    n_lora = sum(p.numel() for _, p in trainable)
    if tuple(sorted(target_modules)) == tuple(sorted(DEFAULT_TARGET_MODULES)) and n_layers == 64:
        assert n_lora == LORA_PARAMS_PER_RANK * lora_r, n_lora
    if lora_compute_dtype == torch.bfloat16:
        for m in peft_model.modules():
            if isinstance(m, LoraLayer):
                m.cast_input_dtype_enabled = False
    peft_model.train()
    peft_model.rltldr_cfg = {"base_dir": str(base_dir), "lora_compute_dtype": lora_compute_dtype,
                             "use_liger": use_liger, "device": device}

    tok = AutoTokenizer.from_pretrained(base_dir)
    if log:
        torch.cuda.synchronize(device)
        print(f"[load_policy] {time.time() - t0:.1f}s (from_pretrained {t_load:.1f}s) | "
              f"{len(expected)} text tensors loaded, 0 missing, skipped {n_visual} visual + {n_mtp} mtp | "
              f"verified {n_verified} tensors bit-exact | LoRA r={lora_r} alpha={lora_alpha}: "
              f"{n_lora / 1e6:.1f}M fp32 params | liger={use_liger} lora_compute={lora_compute_dtype} | "
              f"FLA={fla_mod} | GPU alloc {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB", flush=True)
    return peft_model, tok


# ----------------------------------------------------------------------------------------------------------------
# Log-probs
# ----------------------------------------------------------------------------------------------------------------
class _ChunkedTokenLogProbs(torch.autograd.Function):
    """logp[i] = log_softmax(hidden[i] @ W^T)[target[i]] without materialising more than `chunk` rows of logits.

    Forward saves only (hidden, W, targets, logsumexp). Backward recomputes each chunk's logits and turns them into
    d logp / d logits = onehot(target) - softmax in place, so peak extra memory is ~one fp32 [chunk, V] buffer plus
    its bf16 copy (chunk=2048: 2.0 + 1.0 GB).
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        n = hidden.shape[0]
        out = torch.empty(n, dtype=torch.float32, device=hidden.device)
        lse = torch.empty(n, dtype=torch.float32, device=hidden.device)
        for s in range(0, n, chunk):
            logits = torch.mm(hidden[s:s + chunk], weight.t(), out_dtype=torch.float32)
            lse[s:s + chunk] = torch.logsumexp(logits, dim=-1)
            out[s:s + chunk] = logits.gather(1, targets[s:s + chunk, None]).squeeze(1) - lse[s:s + chunk]
            del logits
        ctx.save_for_backward(hidden, weight, targets, lse)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, targets, lse = ctx.saved_tensors
        chunk = ctx.chunk
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_h = torch.empty_like(hidden) if need_h else None
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        grad_out = grad_out.float()
        for s in range(0, hidden.shape[0], chunk):
            h, g, t = hidden[s:s + chunk], grad_out[s:s + chunk], targets[s:s + chunk]
            buf = torch.mm(h, weight.t(), out_dtype=torch.float32)
            buf.sub_(lse[s:s + chunk, None]).exp_()           # softmax
            buf.mul_(-g[:, None])                               # -g * p
            buf.scatter_add_(1, t[:, None], g[:, None])         # + g * onehot(target)
            dlogits = buf.to(hidden.dtype)
            del buf
            if need_h:
                grad_h[s:s + chunk] = dlogits @ weight
            if need_w:   # lm_head is frozen in this project; kept for completeness
                grad_w.add_(torch.mm(dlogits.t(), h, out_dtype=torch.float32))
            del dlogits
        return grad_h, (grad_w.to(weight.dtype) if need_w else None), None, None


def _backbone_and_head(peft_model):
    causal_lm = peft_model.get_base_model()          # Qwen3_5ForCausalLM with LoRA-wrapped linears
    return causal_lm.model, causal_lm.lm_head.weight


def autocast_ctx(peft_model):
    """Autocast context required around any direct forward of the policy when LoRA runs in bf16."""
    cfg = getattr(peft_model, "rltldr_cfg", {})
    if cfg.get("lora_compute_dtype", torch.float32) == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _offload_ctx(peft_model):
    """Cached TRL activation-offload context manager (it registers hooks on the model, so build it once)."""
    ctx = getattr(peft_model, "_rltldr_offload_ctx", None)
    if ctx is None:
        from trl.models.activation_offloading import get_act_offloading_ctx_manager

        ctx = get_act_offloading_ctx_manager(peft_model, warn_if_no_head=False)
        peft_model._rltldr_offload_ctx = ctx
    return ctx


def token_logprobs(
    peft_model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    chunk: int = 2048,
    offload: bool = False,
    pad_multiple: int = 64,
) -> torch.Tensor:
    """log p(input_ids[t] | input_ids[:t]) for every t in `positions` (fp32, shape [n], differentiable w.r.t. LoRA).

    Args:
        input_ids: LongTensor [L] (one sequence; no packing - the DeltaNet conv has no sequence boundaries).
        positions: LongTensor [n], each 1 <= t < L, any order, duplicates allowed.
        chunk: rows of the vocabulary projection computed at once (2048 -> ~3 GB transient).
        offload: offload per-layer checkpoint inputs to pinned CPU memory (TRL OffloadActivations) during the
            forward; they are copied back during backward. Call .backward() *after* this function returns.
            Measured on GPU 3 (r=32): no time cost at 16k/32k and -19 GiB peak at 32k, so use it for anything
            above ~8k tokens. Safe maxima per micro-step: 32k without offload, 72k with (see README).
        pad_multiple: the backbone only runs on input_ids[:max(positions)] (later tokens cannot influence the
            requested log-probs), right-padded to a multiple of this to limit distinct Triton shapes.
    Uses whatever adapter state is active: wrap in `peft_model.disable_adapter()` for the base model, in
    `adapter_context(...)` for another adapter, and in torch.no_grad() when no gradient is needed.
    """
    backbone, head_w = _backbone_and_head(peft_model)
    device = head_w.device
    ids = input_ids.to(device=device, dtype=torch.long)
    pos = positions.to(device=device, dtype=torch.long)
    if ids.dim() != 1 or pos.dim() != 1:
        raise ValueError("input_ids and positions must be 1-D")
    if pos.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    lo, hi = int(pos.min()), int(pos.max())
    if lo < 1 or hi >= ids.numel():
        raise ValueError(f"positions must lie in [1, {ids.numel() - 1}], got [{lo}, {hi}]")

    ctx_ids = ids[:hi]                                  # hidden state at index t-1 predicts token t
    pad = (-ctx_ids.numel()) % pad_multiple
    if pad:
        ctx_ids = torch.cat([ctx_ids, ctx_ids.new_zeros(pad)])   # causal: padding never affects earlier positions
    with autocast_ctx(peft_model):
        with (_offload_ctx(peft_model) if offload and torch.is_grad_enabled() else nullcontext()):
            hidden = backbone(input_ids=ctx_ids[None], use_cache=False).last_hidden_state[0]
        sel = hidden.index_select(0, pos - 1)
        del hidden
        return _ChunkedTokenLogProbs.apply(sel, head_w, ids[pos], chunk)


# ----------------------------------------------------------------------------------------------------------------
# Adapter export / import
# ----------------------------------------------------------------------------------------------------------------
def _lora_params(peft_model, adapter_name: str) -> dict[str, torch.nn.Parameter]:
    """{export key (vLLM naming): parameter} for one adapter."""
    tag = f".{adapter_name}."
    out = {}
    for n, p in peft_model.named_parameters():
        if (".lora_A." in n or ".lora_B." in n) and tag in n:
            key = n.replace(tag, ".", 1)
            if not key.startswith(TRAINER_LAYER_PREFIX):
                raise AssertionError(f"unexpected LoRA parameter outside decoder layers: {n}")
            out[VLLM_LAYER_PREFIX + key[len(TRAINER_LAYER_PREFIX):]] = p
    if not out:
        raise KeyError(f"adapter {adapter_name!r} has no LoRA parameters")
    return out


def _exchange_paths(a: Path, b: Path) -> bool:
    """Atomically swap two existing paths (Linux renameat2 RENAME_EXCHANGE). False if unsupported."""
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.renameat2
    except (OSError, AttributeError):
        return False
    at_fdcwd, rename_exchange = -100, 2
    if fn(at_fdcwd, os.fsencode(str(a)), at_fdcwd, os.fsencode(str(b)), rename_exchange) == 0:
        return True
    err = ctypes.get_errno()
    if err in (22, 38, 95):   # EINVAL / ENOSYS / EOPNOTSUPP: filesystem or kernel lacks support
        return False
    raise OSError(err, os.strerror(err), str(b))


def _fsync_dir(d: Path) -> None:
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _atomic_dir_write(out_dir: str | os.PathLike, write_fn, overwrite: bool = False) -> str:
    """Write files into a temp dir next to `out_dir` (same filesystem), fsync, then rename into place.

    Readers never observe a partially written directory. With overwrite=True an existing directory is replaced by
    an atomic exchange (renameat2), so `out_dir` always exists and is always complete; on filesystems without
    RENAME_EXCHANGE the old directory is moved aside first (a crash in between leaves `.{name}.old-*`).
    """
    out = Path(out_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (use a new versioned name, or overwrite=True)")
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        write_fn(tmp)
        for f in tmp.iterdir():
            with open(f, "rb") as fh:
                os.fsync(fh.fileno())
        _fsync_dir(tmp)                            # persist the file entries before the rename publishes them
        if out.exists() and _exchange_paths(tmp, out):
            pass                                   # tmp now holds the previous contents; removed below
        elif out.exists():
            old = out.parent / f".{out.name}.old-{uuid.uuid4().hex[:8]}"
            os.rename(out, old)
            os.rename(tmp, out)
            tmp = old
        else:
            os.rename(tmp, out)
        _fsync_dir(out.parent)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # leftover temp dir, or the replaced old contents
    return str(out)


def _adapter_config_dict(peft_model, adapter_name: str) -> dict:
    cfg = peft_model.peft_config[adapter_name].to_dict()
    for k, v in list(cfg.items()):
        if isinstance(v, set):
            cfg[k] = sorted(v)
    cfg["inference_mode"] = True
    return cfg


@torch.no_grad()
def export_adapter(peft_model, out_dir: str | os.PathLike, adapter_name: str = "default",
                   overwrite: bool = False) -> str:
    """Write a vLLM-loadable PEFT adapter (adapter_config.json + adapter_model.safetensors, bf16) atomically.

    Keys are `base_model.model.model.language_model.layers.N.<module>.lora_{A,B}.weight` because vLLM serves the
    multimodal Qwen3_5ForConditionalGeneration (see module docstring). Returns the final directory path.
    """
    cfg = _adapter_config_dict(peft_model, adapter_name)
    if cfg.get("use_dora") or cfg.get("modules_to_save") or cfg.get("bias", "none") != "none":
        raise ValueError("vLLM rejects DoRA / modules_to_save / bias adapters")
    tensors = {k: p.detach().to("cpu", torch.bfloat16).contiguous()
               for k, p in _lora_params(peft_model, adapter_name).items()}

    def write(tmp: Path):
        save_file(tensors, str(tmp / ADAPTER_WEIGHTS), metadata={"format": "pt"})
        (tmp / ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2))

    return _atomic_dir_write(out_dir, write, overwrite=overwrite)


@torch.no_grad()
def load_adapter_weights(peft_model, adapter_dir: str | os.PathLike, adapter_name: str = "default") -> str:
    """Load an adapter written by export_adapter (inverse key rename, bf16 -> fp32) into `adapter_name`.

    If `adapter_name` does not exist yet it is created from the saved adapter_config.json and kept frozen
    (requires_grad=False) - use `adapter_context(peft_model, adapter_name)` to compute log-probs under it, and
    `peft_model.delete_adapter(adapter_name)` to free it (~0.9 GB fp32 at r=32). Loading into "default" overwrites
    the trainable weights in place (optimizer state is untouched). Strict: every tensor must match exactly one
    LoRA parameter of that adapter, with the same shape.
    """
    from peft import LoraConfig

    adapter_dir = Path(adapter_dir)
    saved_cfg = json.loads((adapter_dir / ADAPTER_CONFIG).read_text())
    if adapter_name not in peft_model.peft_config:
        keep = {k: saved_cfg[k] for k in ("r", "lora_alpha", "target_modules", "lora_dropout", "bias",
                                          "use_rslora", "use_dora") if k in saved_cfg}
        flags = {n: p.requires_grad for n, p in peft_model.named_parameters()}
        peft_model.add_adapter(adapter_name, LoraConfig(task_type="CAUSAL_LM", **keep))
        for n, p in peft_model.named_parameters():   # restore the trainable set; the new adapter stays frozen
            p.requires_grad_(flags.get(n, False))
        if getattr(peft_model, "rltldr_cfg", {}).get("lora_compute_dtype") == torch.bfloat16:
            from peft.tuners.lora import LoraLayer
            for m in peft_model.modules():
                if isinstance(m, LoraLayer):
                    m.cast_input_dtype_enabled = False
    cur = peft_model.peft_config[adapter_name]
    for k in ("r", "lora_alpha"):
        if saved_cfg.get(k) != getattr(cur, k):
            raise ValueError(f"adapter {k}={saved_cfg.get(k)} != model adapter {adapter_name!r} {k}={getattr(cur, k)}")
    if bool(saved_cfg.get("use_rslora", False)) != bool(cur.use_rslora):
        raise ValueError("use_rslora mismatch")

    params = _lora_params(peft_model, adapter_name)
    sd = load_file(str(adapter_dir / ADAPTER_WEIGHTS))
    missing, unexpected = sorted(set(params) - set(sd)), sorted(set(sd) - set(params))
    if missing or unexpected:
        raise KeyError(f"adapter key mismatch: missing {missing[:5]} ({len(missing)}), "
                       f"unexpected {unexpected[:5]} ({len(unexpected)})")
    for k, p in params.items():
        t = sd[k]
        if t.shape != p.shape:
            raise ValueError(f"shape mismatch for {k}: {tuple(t.shape)} vs {tuple(p.shape)}")
        p.copy_(t.to(device=p.device, dtype=p.dtype))
    return adapter_name


@contextmanager
def adapter_context(peft_model, adapter_name: str):
    """Temporarily make `adapter_name` the only active adapter (frozen); restores the previous active adapter(s)
    and every parameter's requires_grad flag on exit."""
    prev = peft_model.base_model.active_adapter
    flags = {n: p.requires_grad for n, p in peft_model.named_parameters()}
    peft_model.base_model.set_adapter(adapter_name, inference_mode=True)
    try:
        yield peft_model
    finally:
        peft_model.base_model.set_adapter(prev)
        for n, p in peft_model.named_parameters():
            p.requires_grad_(flags[n])


# ----------------------------------------------------------------------------------------------------------------
# Trainer state (crash recovery)
# ----------------------------------------------------------------------------------------------------------------
TRAINER_LORA = "lora_fp32.safetensors"
TRAINER_OPT = "optimizer.pt"
TRAINER_META = "meta.json"


def save_trainer_state(peft_model, optimizer: torch.optim.Optimizer | None, out_dir: str | os.PathLike,
                       metadata: dict | None = None, extra: dict | None = None, adapter_name: str = "default",
                       overwrite: bool = True) -> str:
    """Atomically write LoRA fp32 weights (trainer naming) + optimizer state + JSON metadata.

    By default an existing `out_dir` is replaced atomically (one rolling crash-recovery checkpoint); readers
    always see either the previous or the new complete checkpoint.

    `metadata` must be JSON-serialisable (policy version, step, data cursor, ...); `extra` is any torch.save-able
    dict (e.g. LR-scheduler or RNG state). The optimizer must have been built over this model's parameters.
    """
    tag = f".{adapter_name}."
    lora = {n: p for n, p in peft_model.named_parameters() if ("lora_A" in n or "lora_B" in n) and tag in n}
    tensors = {n: p.detach().to("cpu", torch.float32).contiguous() for n, p in lora.items()}
    id2name = {id(p): n for n, p in peft_model.named_parameters()}
    opt_blob = None
    if optimizer is not None or extra is not None:   # `extra` lives in the same file, also without an optimizer
        order = None if optimizer is None else [id2name[id(p)] for g in optimizer.param_groups for p in g["params"]]
        opt_blob = {"optimizer": None if optimizer is None else optimizer.state_dict(), "param_names": order,
                    "extra": extra}
    meta = {"saved_at": time.time(), "adapter_name": adapter_name, "n_tensors": len(tensors),
            "lora_config": _adapter_config_dict(peft_model, adapter_name),
            "torch": torch.__version__, "metadata": metadata or {}}

    def write(tmp: Path):
        save_file(tensors, str(tmp / TRAINER_LORA))
        if opt_blob is not None:
            torch.save(opt_blob, tmp / TRAINER_OPT)
        (tmp / TRAINER_META).write_text(json.dumps(meta, indent=2, default=str))

    return _atomic_dir_write(out_dir, write, overwrite=overwrite)


@torch.no_grad()
def load_trainer_state(peft_model, optimizer: torch.optim.Optimizer | None, ckpt_dir: str | os.PathLike,
                       adapter_name: str = "default") -> tuple[dict, dict | None]:
    """Restore what save_trainer_state wrote. Returns (metadata, extra)."""
    ckpt = Path(ckpt_dir)
    meta = json.loads((ckpt / TRAINER_META).read_text())
    if meta["adapter_name"] != adapter_name:
        raise ValueError(f"checkpoint holds adapter {meta['adapter_name']!r}, asked for {adapter_name!r}")
    cur = _adapter_config_dict(peft_model, adapter_name)
    for k in ("r", "lora_alpha", "target_modules"):
        if meta["lora_config"].get(k) != cur.get(k):
            raise ValueError(f"LoRA config mismatch on {k}: ckpt {meta['lora_config'].get(k)} vs model {cur.get(k)}")
    sd = load_file(str(ckpt / TRAINER_LORA))
    tag = f".{adapter_name}."
    params = {n: p for n, p in peft_model.named_parameters() if ("lora_A" in n or "lora_B" in n) and tag in n}
    if set(sd) != set(params):
        raise KeyError(f"LoRA key mismatch: {len(set(params) - set(sd))} missing, {len(set(sd) - set(params))} extra")
    for n, p in params.items():
        p.copy_(sd[n].to(device=p.device, dtype=p.dtype))
    blob = None
    if optimizer is not None or (ckpt / TRAINER_OPT).exists():   # optimizer requested -> the file must exist
        blob = torch.load(ckpt / TRAINER_OPT, map_location="cpu", weights_only=False)
    if optimizer is not None:
        if blob.get("optimizer") is None:
            raise ValueError(f"{ckpt} was saved without optimizer state")
        id2name = {id(p): n for n, p in peft_model.named_parameters()}
        order = [id2name[id(p)] for g in optimizer.param_groups for p in g["params"]]
        if order != blob["param_names"]:
            raise ValueError("optimizer parameter order differs from the checkpoint")
        optimizer.load_state_dict(blob["optimizer"])   # moves state tensors to each param's device
    return meta["metadata"], (None if blob is None else blob.get("extra"))
