# Qwen3.8-27B Natural Language Autoencoder (layer 42)

**Problem.** Natural Language Autoencoders (NLAs) turn a model's internal activations into readable
text. A verbalizer (AV) writes an explanation of an activation, and a reconstructor (AR) must rebuild
the activation from that text alone. Qwen3.8-27B had no NLA. The public Qwen3.6-27B NLA does not
transfer to it: unchanged on 3.8, it produced 0% parseable explanations (see `RESULTS.md`).

**Approach.** We use the EasyNLA recipe on Qwen3.8-27B at residual-stream layer 42 of 64.

1. Re-extract Qwen3.8 activations for 146k text prefixes. Each prefix has a public gold explanation
   written by Claude Sonnet 4.6.
2. Warm-start the AV and the AR with supervised fine-tuning (SFT). Both are LoRA r64 on the 3.8 base.
   The AR uses blocks 0–42 and a linear head.
3. Run GRPO. The reward is the negative reconstruction MSE, and the critic is co-trained.

All GPU work runs on Modal. Training is data parallel over B200s.

- AV: the activation is added at a marker token after block 1. The norm is matched to the residual.
  Then the AV writes an `<explanation>`.
- AR: the explanation text goes to a `Linear(5120, 5120)` head, which outputs an activation.
  Metric: fraction of variance explained (FVE).

## Layout

| file | what |
|---|---|
| `modal_nla38.py` | full pipeline: transfer test, data, AV/AR SFT, GRPO, eval, export (see docstring) |
| `nla_qwen38.py` | standalone inference: `extract` / `explain` / `reconstruct` / `fve` |
| `patch_qwen35_gva_518.py` | transformers 5.18 patch: grouped-head fla kernel for B200 training |
| `spawn.py`, `status.sh` | spawn jobs on the deployed Modal app; tail live container logs |
| `demo/` | NLA Explorer web demo: click any token and read its explanation |
| `README_HF.md` | model card for the HF repo |
| `RESULTS.md` | each measured number, with split, baseline and command |

## Commands

```bash
cd nla
modal deploy modal_nla38.py                                      # training functions
python spawn.py train_rl '{"max_usd": 50}'                        # e.g. GRPO with a $ cap
modal run modal_nla38.py::evaluate --tag eval_sft                 # held-out FVE + shuffled control
modal run demo/demo_app.py::prefetch && modal deploy demo/demo_app.py   # live demo (prints the URL)
```

Inference without Modal (one 141–180 GB GPU or two GPUs):

```python
from nla_qwen38 import NLA
nla = NLA.load("gereon/qwen3.8-27b-nla-L42", stage="sft")      # private repo; "rl" when uploaded
acts = nla.extract(["The quick brown fox jumps over the"])
print(nla.explain(acts)[0])
```

## Artifacts

- Model (private): `gereon/qwen3.8-27b-nla-L42` on Hugging Face: AV/AR LoRAs, config and inference script.
- Data (private): `gereon/qwen3.8-27b-nla-L42-data`: extracted activations, row pools and eval JSONs.

## Credits

- **Method:** Natural Language Autoencoders (Anthropic, 2026). Activation injection follows
  Karvonen et al. (Activation Oracles).
- **Code used as a runtime dependency (cloned, not copied):** [EasyNLA](https://github.com/asherps/EasyNLA)
  (MIT), with [ceselder's fork](https://github.com/ceselder/EasyNLA) at commit `d23cba0` for qwen3_5
  support. Both are based on [nanoNLA](https://github.com/ceselder/nanoNLA). The GVA patch is our port
  of the idea in ceselder's `utils/patch_transformers_qwen35_gva.py` to transformers 5.18.
- **Models:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
  [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (Apache-2.0). We used
  [ceselder/qwen3.6-27b-nla-L42](https://huggingface.co/ceselder/qwen3.6-27b-nla-L42) only for the
  transfer test, and its tokenizer and prompt template as the NLA prompt contract.
- **Data:** [ceselder/qwen3-8b-nla-L24-finefineweb-100k](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k).
  It contains FineFineWeb text with Claude Sonnet 4.6 gold explanations. We re-extract the activations.
- **Libraries:** PyTorch, transformers, PEFT, safetensors, flash-linear-attention, causal-conv1d,
  huggingface_hub, pyarrow, FastAPI (demo).
- **Compute:** Modal (B200).
