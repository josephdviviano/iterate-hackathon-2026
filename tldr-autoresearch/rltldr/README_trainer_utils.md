# Trainer model utilities (`rltldr/model_utils.py`)

These are the building blocks the policy trainer on GPU 3 uses for Qwen3.8-27B:

- load the frozen bf16 base and attach a LoRA;
- compute per-token log-probs without building the full logits tensor;
- export an adapter that vLLM can load (with key names vLLM really applies);
- save and restore checkpoints so a crashed trainer can resume.

Everything below was measured on 2026-10-03:

- **Trainer:** GPU 3 (RTX PRO 6000 Blackwell, sm_120), torch 2.13.0+cu130, transformers 5.18.0, peft 0.21.2, liger-kernel 0.8.4, FLA 0.5.2.
- **Server:** the live vLLM 0.30.0 server (`qwen3.8-27b-fp8`, TP=2).

Import with `from rltldr import model_utils as mu`. The package `__init__` is left empty on purpose, so importing the package does not pull in torch.

## API

```python
pm, tok = mu.load_policy(base_dir, lora_r=32, lora_alpha=32, target_modules=mu.DEFAULT_TARGET_MODULES,
                         device="cuda", gradient_checkpointing=True, use_liger=True,
                         lora_compute_dtype=torch.bfloat16, verify_weights="spot")
lp = mu.token_logprobs(pm, input_ids, positions, chunk=2048, offload=False)   # FloatTensor[n], differentiable
mu.export_adapter(pm, "data/adapters/policy-v7")                              # vLLM-loadable, atomic
mu.load_adapter_weights(pm, adapter_dir, adapter_name="default")              # inverse of export_adapter
with mu.adapter_context(pm, "old"): ...                                       # compute under another adapter
mu.save_trainer_state(pm, opt, ckpt_dir, metadata={...}, extra=None)          # LoRA fp32 + AdamW + JSON meta
meta, extra = mu.load_trainer_state(pm, opt, ckpt_dir)
mu.verify_base_weights(pm, base_dir, names=None)                              # bit-exact check vs checkpoint
mu.autocast_ctx(pm)                                                           # needed for any direct forward
```

### `load_policy`

**What it loads.**
- Builds `Qwen3_5ForCausalLM` from `config.text_config` of the dequantized checkpoint (config `trainer_base_dir`, default `~/models/Qwen3.8-27B-FP8-dequant-bf16`, made by `tools/dequant_fp8.py`).
- transformers maps the keys `model.language_model.*` to `model.*`. `lm_head.weight` keeps its name and is untied.
- Settings: SDPA attention, `use_cache=False`, non-reentrant gradient checkpointing.
- The model is returned in `train()` mode. Checkpointing only runs in that mode.

**Load-time checks.** Each one is an assert, so a bad load stops immediately.
- `missing_keys`, `mismatched_keys` and `error_msgs` must all be empty.
- `unexpected_keys` may contain only visual or MTP keys.
- The model's state-dict key set must equal the checkpoint index's text keys exactly.
- `verify_weights="spot"` compares 8 tensors with the checkpoint bit for bit. `"full"` compares every tensor.
- transformers' DeltaNet kernel must resolve to `fla.ops.gated_delta_rule.chunk` and not to the torch fallback.
- The trainable parameters must be exactly the LoRA A/B matrices, all in fp32.
- For the default targets, the LoRA must have 6,799,360·r parameters.

**Liger.** Liger RMSNorm and SwiGLU are patched in at class level before the model is built. RoPE and fused CE are not used.

**LoRA compute dtype.**
- Default: master weights stay in fp32, and the LoRA matmuls run in bf16 under `torch.autocast`, with PEFT's fp32 up-cast of the inputs switched off. vLLM also applies LoRA in bf16.
- Autocast leaves the base model unchanged: base log-probs with and without autocast are bit-identical (max |Δ| = 0).
- Consequence: **a direct `pm(...)` call needs `with mu.autocast_ctx(pm):`**. Everything inside `model_utils` already does this.

### `token_logprobs`

Returns `log p(ids[t] | ids[:t])` for each `t` in `positions` (1 ≤ t < L, in any order).

How it works:
- The backbone runs only on `ids[:max(positions)]`, right-padded to a multiple of 64. Padding is causal, so the results do not change.
- The final hidden states then go through `_ChunkedTokenLogProbs`, a custom autograd function:
  - **Forward:** a bf16 GEMM with fp32 output, so the logits are not rounded to bf16. It is done in chunks of `chunk` rows. It stores only the hidden states, the targets and the logsumexp.
  - **Backward:** recomputes each chunk and builds `onehot − softmax` in place. Peak extra memory is about 3 GB at chunk 2048.
  - The full [L, 248320] logits tensor never exists.

What it follows:
- It uses whichever adapter is currently active:
  - base model: `pm.disable_adapter()`;
  - another adapter: `adapter_context(pm, name)`;
  - wrap in `torch.no_grad()` for old-policy or behaviour recomputes.
- `offload=True` applies TRL's `OffloadActivations` to the backbone forward. This moves the per-layer checkpoint inputs to pinned CPU memory. Call `.backward()` after the function returns.

### Adapters

`export_adapter(pm, out_dir, adapter_name="default", overwrite=False)` writes:
- `adapter_config.json`: the PEFT config, with sets turned into lists and `inference_mode` set to true;
- `adapter_model.safetensors`: bf16 tensors keyed `base_model.model.model.language_model.layers.N.<mod>.lora_{A,B}.weight`.

That key name matters. vLLM serves the multimodal class and maps `model.language_model.` to `language_model.model.`. Plain PEFT keys (`...model.layers.N`) match no module, and the adapter then silently does nothing.

Writing is atomic:
- Files go to a temp directory in the same parent, are fsynced, then renamed into place.
- By default the call refuses to overwrite: use a new versioned name for each policy version.
- With `overwrite=True` the existing directory is swapped atomically with `renameat2(RENAME_EXCHANGE)`. I checked that this works on the overlay filesystem here.

`load_adapter_weights(pm, dir, adapter_name)`:
- Strictly reverses the rename and copies bf16 into fp32.
- Every file key must match exactly one LoRA parameter of that adapter, with the same shape.
- Into `"default"`, it overwrites the trainable weights in place; optimizer state is left alone.
- Into a new name, it creates a frozen second adapter. Use it through `adapter_context` to recompute log-probs under an older policy. `pm.delete_adapter(name)` frees it (about 0.9 GB at r=32).

### Trainer state

`save_trainer_state(pm, opt, dir, metadata, extra=None, overwrite=True)` writes:
- `lora_fp32.safetensors` (trainer naming, fp32);
- `optimizer.pt`: the optimizer `state_dict`, the parameter names in optimizer order, and `extra` (also written for `opt=None` when `extra` is given);
- `meta.json`: your metadata plus the LoRA config.

It acts as a single rolling checkpoint: the default `overwrite=True` swaps the directory atomically.

`load_trainer_state` checks before loading:
- the adapter name;
- r, alpha and target_modules;
- the exact set of LoRA keys;
- that the optimizer's parameter order matches the saved one.

## Measured results

### (a) Loading

The checkpoint files were in the page cache for these runs.

| Item | Value |
|---|---|
| `load_policy` total | 8.4–15.8 s; `from_pretrained` alone 4.3–4.8 s |
| GPU memory after load | 50.91 GiB allocated, 55.17 GiB reserved, 55.8 GiB in use on the device |
| Text tensors loaded | 851 |
| Missing / unexpected / mismatched keys | 0 / 0 / 0 |
| Checkpoint keys deliberately skipped | 333 `model.visual.*` + 15 `mtp.*` |
| Full bit-exact verification | all 851 tensors equal the checkpoint, in 4.7 s |
| Static memory with LoRA + grads + AdamW state | 53.41 GiB |

### (e) FLA DeltaNet kernel on sm_120

Script: `tests/trainer/test_fla_sm120.py`. Shapes: 48 heads × 128, q/k L2-normalised in the kernel, fp32 g, bf16 beta. Result: **PASS**.

- Forward relative error vs the fp32 naive recurrence: 4.06e-3.
- Gradient relative errors: q 4.39e-3, k 4.70e-3, v 4.09e-3, g 3.80e-3, beta 4.09e-3.
- No NaN or Inf at T = 8k, 32k or 64k.
- Timing at 64k: forward 14.5 ms, backward 48 ms.

### (b) Parity with the live vLLM server

Script: `tests/trainer/validate_parity.py`.

**Test sequence.**
- 1565 tokens: a 165-token chat prompt rendered with the tokenizer's chat template, plus a 1400-token completion sampled from the served base model at temperature 1.
- The trainer's template rendering equals vLLM's `prompt_token_ids` exactly.
- vLLM log-probs come from `/v1/completions` with `prompt_logprobs:0`. These requests bypass the prefix cache. For the **base model** (and a B=0 adapter) they are deterministic: a repeat gives max |Δ| = 0. With a non-zero LoRA they are **not** (see below).

**Base model.**

| Trainer vs vLLM (base) | Mean \|Δ\| | Median | p99 | Mean signed |
|---|---|---|---|---|
| All 1564 positions | 0.0777 | 0.0064 | 1.23 | +0.031 |
| Completion tokens (on-policy; mean log p −0.56) | **0.0294** | 0.0044 | 0.24 | +0.0002 |
| Prompt tokens (system/user text; mean log p −6.97) | 0.490 | 0.40 | 2.03 | +0.30 |

The mismatch grows as tokens get less likely:

| vLLM log p range | Mean \|Δ\| |
|---|---|
| > −0.05 | 0.002 |
| −0.5 to −0.05 | 0.023 |
| −2 to −0.5 | 0.066 |
| −5 to −2 | 0.205 |
| < −10 | 0.77 |

**Adapter with random B** (B ~ N(0, 0.01), A = PEFT init), exported as `data/adapters/parity-test-1` and loaded as `parity-test-1-<ts>`:
- Trainer vs vLLM: mean |Δ| 0.111 over all positions, **0.0304 on completion tokens**.
- vLLM's adapter log-probs differ from its base log-probs, so the naming works.
- Mean |base − adapter| gap over all positions: 0.093 in the trainer, 0.135 in vLLM; per-token correlation 0.69. On completion tokens only: 0.048 vs 0.058.
- Timings: export 0.7–5.6 s (435 MB, mostly fsync); `/v1/load_lora_adapter` 0.09–0.2 s; first request 0.24 s.

**Why the random-B gap disagrees between the two systems.** It is a noise floor, not a naming bug.
- With B ~ N(0, 1e-4) (`--mode random`), the weight change is negligible.
- Even so, the per-token log-probs move by a mean |Δ| of 0.077–0.093 in vLLM and 0.020 in the trainer.
- Acceptance re-test (2026-10-03): most of vLLM's part is **run-to-run nondeterminism of its LoRA path**, not a deterministic re-rounding. The same loaded `parity-test-1`, queried twice with identical requests, gives mean |Δ| 0.070 over all positions (completion tokens 0.037, p99 0.31, max 2.9–5.0); two loads of the same adapter differ by 0.078. A greedy decode under that adapter diverged from its own repeat after 13 of 64 tokens. The base model and a B=0 adapter are bit-deterministic. Likely cause: non-deterministic accumulation in vLLM's Triton LoRA kernels (`_lora_shrink_kernel`), amplified by per-token FP8 activation quantisation. The re-test's random-B effect correlation was 0.58 (vs 0.69 here) for the same reason.
- A random adapter with mean effect ~0.1 is therefore buried in vLLM's noise.
- Mean shifts over the completion tokens stay tiny (|Δ| < 0.004).

**Decisive test: a trained adapter transfers exactly** (`tests/trainer/validate_adapter_modules.py`).

Method: for each module type alone, and for all of them together, fit a LoRA for 2–5 Adam steps to raise the completion log-likelihood. Export it, load it into vLLM, and compare the resulting shift in both systems.

| Adapter on | Mean completion log p shift: trainer | vLLM | Per-token effect corr | Mismatch after (completion) |
|---|---|---|---|---|
| all 10 types | +0.4842 | +0.4843 | 0.996 | 0.004 |
| q_proj | +0.4342 | +0.4341 | 0.995 | 0.014 |
| k_proj | +0.2978 | +0.2966 | 0.991 | 0.018 |
| v_proj | +0.3197 | +0.3196 | 0.994 | 0.020 |
| o_proj | +0.4309 | +0.4279 | 0.995 | 0.013 |
| gate_proj | +0.3770 | +0.3792 | 0.997 | 0.014 |
| up_proj | +0.3505 | +0.3499 | 0.997 | 0.015 |
| down_proj | −0.0671 (overshot) | −0.0648 | 0.998 | 0.034 |
| in_proj_qkv | +0.2536 | +0.2548 | 0.996 | 0.027 |
| in_proj_z | +0.4673 | +0.4655 | 0.993 | 0.018 |
| out_proj | +0.2960 | +0.2932 | 0.995 | 0.021 |

So vLLM (TP=2, FP8 base) applies every module's LoRA, including the packed `in_proj_qkvz` and `qkv_proj` splits. Shifts agree to within 0.003 nat/token.

**Other parity facts:**
- In vLLM, a B=0 adapter gives log-probs bit-identical to the base model name, so a zero `policy-v0` is exactly the base.
- Liger vs no Liger: base parity with vLLM is unchanged (completion 0.0294 vs 0.0292). Trainer-vs-trainer mean |Δ| is 0.0196, which is the bf16 noise floor.

### (d) Round trip

Both checks give bit-identical log-probs (max |Δ| = 0):
- `export_adapter` → zero the LoRA → `load_adapter_weights` into `default`.
- Loading the same adapter into a second slot `old` and computing under `adapter_context`.

Afterwards, `default` and every `requires_grad` flag are restored. With the default's B set to zero, the output equals the base exactly.

### (c) One training micro-step on the full 64-layer model

Script: `tests/trainer/bench_train_step.py`. One micro-step is:
- forward with grad on the LoRA (r=32, all 10 module types);
- `token_logprobs` over all L−1 positions and a weighted loss;
- backward.

AdamW state is resident throughout. "s" is the second call at that length. Usable device memory is about 94.4 GiB.

| L | Offload | s | tok/s | Peak alloc GiB | Peak reserved GiB | Result |
|---|---|---|---|---|---|---|
| 8k | no | 5.67 | 1445 | 62.8 | 63.0 | ok |
| 16k | no | 11.94 | 1373 | 69.3 | 70.6 | ok |
| 24k | no | 18.62 | 1320 | 77.3 | 79.1 | ok |
| 32k | no | 25.70 | 1275 | 85.3 | 87.7 | **max safe without offload** |
| 36k | no | 29.44 | 1252 | 89.3 | 91.8 | ok (2.6 GiB headroom) |
| 40k | no | 33.47 | 1224 | 93.2 | 93.8 | ran only after allocator retries |
| 44k | no | — | — | — | — | **OOM** |
| 16k | yes | 11.93 | 1374 | 59.8 | 61.8 | ok (no time cost) |
| 32k | yes | 25.71 | 1275 | 66.2 | 70.1 | ok (no time cost, −19 GiB) |
| 48k | yes | 41.31 | 1190 | 72.6 | 78.5 | ok |
| 56k | yes | 49.91 | 1149 | 75.8 | 83.1 | ok |
| 64k | yes | 58.77 | 1115 | 79.0 | 86.3 | ok |
| 72k | yes | 68.37 | 1078 | 82.2 | 90.7 | **max safe with offload** |
| 80k | yes | 81.16 | 1009 | 85.4 | 93.5 | ran, at the limit |
| 88k | yes | 91.38 | 986 | 88.6 | 93.6 | ran, at the limit |

Other measurements:
- **No-grad forward** (for old-policy or behaviour log-probs): 32k 7.5 s (61.8 GiB peak); 64k 16.2 s (68.1 GiB).
- **First call at a new, larger length** can take an extra 13–14 s, e.g. 24k: 31 s vs 19 s. This is one-off allocator growth or autotune, not repeated.
- **Liger on vs off** (`bench_noliger_bf16.json`): 16k 11.94 s vs 13.44 s; 32k 25.70 s vs 28.65 s (about 10% faster). Peak allocated memory is 1.0 GiB lower at 16k and 2.0 GiB lower at 32k. Liger is on by default.
- **bf16 vs fp32 LoRA compute** (`bench_liger_fp32lora.json`): 16k 11.94 s vs 15.16 s; 32k 25.70 s vs 31.99 s (about 20% faster). Peak allocated memory is 4.6 GiB lower at 32k. bf16 is the default.

**Recommendations for the trainer:**
- Pass `offload=True` for any sequence longer than about 8k. It costs no time and saves memory.
- Cap the segments of one micro-step at 72k tokens.
- Keep `chunk=2048`.

### Tiny-model functional tests

Script: `tests/trainer/test_model_utils_tiny.py`, run on a 4-layer random composite checkpoint that includes visual and MTP keys. Result: **28/28 PASS** (re-run 3x after the acceptance fixes). Covered:
- the loading asserts;
- chunked log-probs vs an fp32-logit reference: max |Δ| 1.9e-6;
- LoRA gradient accuracy vs an all-fp32 copy of the model: our relative error 2.37e-2 vs 2.42e-2 for HF's own full-logits bf16 path;
- d/dhidden of the custom function: relative error 2.8e-3;
- trim and pad, autocast neutrality, and offload equivalence (values identical; gradients within run-to-run kernel noise);
- export keys, dtype, config, atomicity and refusal to overwrite;
- round trips, the second adapter and `adapter_context`;
- trainer-state round trip: weights and AdamW state restored bit-exactly, identical next AdamW step given the same gradients (the backward itself is not bit-deterministic, so the test replays recorded gradients), atomic overwrite, and `extra` kept for a weights-only (optimizer=None) checkpoint.

## Running the tests

All tests run on the trainer GPU only (GPU 3 on our machine). Use the train env, from the project root:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(~/envs/serve/bin/python -m rltldr.config get trainer_gpu_uuid)
PY=~/envs/train/bin/python; T=tests/trainer
$PY $T/test_fla_sm120.py                  # (e) ~30 s
$PY $T/test_model_utils_tiny.py           # functional, ~1 min
$PY $T/validate_parity.py                 # (a)(b)(d) vs live vLLM, ~1 min; unloads its adapter
$PY $T/validate_adapter_modules.py        # trained per-module transfer check, ~3 min
$PY $T/bench_train_step.py                # (c) ~20 min
```

Results land in `data/parity/*.json` and `data/bench/*.json`; logs in `logs/`.

## Open issues and caveats

- **Per-token noise between vLLM and the trainer.**
  - On completion tokens the mean |Δ| is about 0.03, but it is heavy-tailed: p99 0.24, max 2.5.
  - Low-probability tokens differ by 0.2–0.8 nat.
  - vLLM's per-token log-probs under any non-zero LoRA are nondeterministic run to run (identical request repeated: mean |Δ| 0.037 on completion tokens, p99 0.31; greedy decodes diverge). So behaviour log-probs returned by vLLM for a policy-vN (N ≥ 1) carry that noise on top of the trainer/vLLM mismatch; a serving-side fix would need deterministic LoRA kernels (e.g. vLLM batch-invariant mode or split-K = 1 LoRA kernel configs), which requires a server restart.
  - Implications:
    - per-token truncated-IS ratios are noisy and need the clamp;
    - track the mean |Δ| on action tokens, not the max;
    - compare policy versions with aggregate statistics.
- **Prompt-region offset.** On this sequence the vLLM *base* run gives system/user tokens log-probs 0.30 nat lower than the trainer. Any LoRA, even B ~ 1e-4, raises vLLM's prompt-region mean by 0.11–0.41 nat, so the offset looks like one unlucky FP8 noise realisation and not a systematic bug. It does not affect action tokens (mean signed difference +0.0002). I checked only one sequence.
- **causal-conv1d is not installed**, so transformers uses its torch `F.conv1d` fallback. This is included in all the timings above.
- **Backward is not bit-deterministic** (FLA/SDPA kernels): run-to-run relative difference in the LoRA gradients is about 1e-3 on the tiny model.
- **Liger patching is process-global.** `load_policy(use_liger=False)` restores the original classes. Do not mix both settings in one process with models alive.
- **Rolling checkpoint.** `save_trainer_state` overwrites by default. Keep versioned adapters separately with `export_adapter`, which refuses to overwrite.
