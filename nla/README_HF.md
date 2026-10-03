---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- interpretability
- natural-language-autoencoder
- nla
- activation-verbalizer
- qwen3_5
language:
- en
---

# Qwen3.8-27B Natural Language Autoencoder (layer 42)

A **Natural Language Autoencoder** (NLA — [Anthropic, 2026](https://transformer-circuits.pub/2026/nla/))
for **Qwen/Qwen3.8-27B** at residual-stream **layer 42/64**, trained with the EasyNLA recipe
(AV/AR warm-start SFT on gold explanations → GRPO with a co-trained critic).

> **Status: work in progress.** This repo is updated as training proceeds; see
> `eval_*.json` and the table below for the checkpoint currently uploaded.

- **AV (verbalizer)** — LoRA (r64, rsLoRA, all linear layers) on Qwen3.8-27B. An activation is
  added, norm-matched, at the marker token `㈜` after block 1, and the AV writes an
  `<explanation>…</explanation>`.
- **AR (reconstructor)** — Qwen3.8-27B truncated to blocks 0–42 (+LoRA r64, final norm removed)
  and a `Linear(5120, 5120)` head that maps the explanation back to the activation.

## Why not reuse the Qwen3.6-27B NLA?

[`ceselder/qwen3.6-27b-nla-L42`](https://huggingface.co/ceselder/qwen3.6-27b-nla-L42) does **not**
transfer: Qwen3.8 shares the architecture/tokenizer but its weights moved 30–60% (relative
Frobenius) from 3.6; layer-42 activations have centered cosine 0.82 to 3.6's. Run unchanged on
Qwen3.8, the 3.6 verbalizer produced **0%** parseable explanations (vs 75.5% FVE on 3.6 in the
same harness).

## Results (held-out, doc-disjoint)

| checkpoint | held-out FVE | shuffled-activation control |
|---|---|---|
| SFT warm-start (AV 1 epoch + AR 1 epoch) | **53.2%** (99.6% well-formed) | −76.3% |
| GRPO | _in progress_ | |

Held-out = 256 rl-split prefixes whose docs never appear in training (doc-hash bucket). Greedy decoding; scored by the matching AR. AR alone on gold Claude explanations: 58.4% (val).

FVE = 1 − MSE(n(pred), n(gold)) / Var(n(gold)), n = rescale to norm √d (paper definition).

## Usage

```python
from nla_qwen38 import NLA          # nla_qwen38.py is in this repo
nla = NLA.load("gereon/qwen3.8-27b-nla-L42", stage="rl")   # "sft" for the warm-start
acts = nla.extract(["The quick brown fox jumps over the"])    # [N, 5120] layer-42, last token
print(nla.explain(acts)[0])
recon = nla.reconstruct(nla.explain(acts))
print(nla.fve(recon, acts))
```

Needs `transformers>=5.5` (qwen3_5), `peft`, `safetensors`; install `flash-linear-attention` and
`causal-conv1d` for usable speed. AV ≈ 54 GB + AR ≈ 36 GB in bf16 (`ar_device=` to split GPUs).

## Contents

| path | what |
|---|---|
| `av_sft_lora/` | verbalizer warm-start LoRA (PEFT) |
| `av_rl_lora/` | verbalizer GRPO LoRA, stacked on top of `av_sft_lora` |
| `ar_sft_critic/` | reconstructor warm-start (LoRA + value head, `ar_lora_value_head.safetensors`) |
| `rl_critic/` | reconstructor co-trained through RL (the reward model) |
| `nla_config.json` | exact rendered AV prompt + token ids, marker id, critic template |
| `nla_meta.yaml` | EasyNLA-style sidecar |
| `nla_qwen38.py` | standalone inference (extract / explain / reconstruct / fve) |

## Training data & recipe

- Texts + Claude Sonnet 4.6 gold explanations from
  [`ceselder/qwen3-8b-nla-L24-finefineweb-100k`](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k);
  activations re-extracted from Qwen3.8-27B at layer 42 (last token of each prefix).
- 120k (text, explanation) rows (45.8k docs) for both AV and AR SFT, one epoch, batch 64, lr 1e-4.
- GRPO on 25k RL-split prefixes: reward = −reconstruction MSE, group size 8, k3 KL (β=0.01) to the
  SFT policy, hinged length penalty, critic co-trained every step.
- Eval docs are doc-hash disjoint from all training docs.
- Code: `modal_nla38.py` (Modal, data-parallel B200s), built on
  [EasyNLA](https://github.com/asherps/EasyNLA) / [ceselder's fork](https://github.com/ceselder/EasyNLA).
