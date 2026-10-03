# Results: Qwen3.8-27B NLA, layer 42

## Definitions

- **FVE.** 1 − MSE(n(pred), n(gold)) / mean((n(gold) − mean n(gold))²). The function n rescales a
  vector to norm √5120. The 0% FVE baseline is "predict the mean activation".
- **Held-out eval set.** 256 prefixes from the RL split. Their docs have crc32(doc_id) % 1000 < 5,
  and no doc in this bucket is in any training split. This includes ceselder's 3.6 training: the av,
  ar and rl splits are doc-disjoint, and RL val buckets ≥ 5.
- **Val set.** 1,024 prefixes in doc buckets [5, 25). It is disjoint from the train pool, which uses
  buckets ≥ 25.
- **Number of runs.** Every result comes from one run (seed 0). Decoding is greedy unless noted.

## Table

| # | result | metric | split | baseline | command |
|---|---|---|---|---|---|
| 1 | ceselder's 3.6 NLA on **Qwen3.6** (harness check) | **75.5% FVE**, 80% extraction | held-out 256 | ceselder reports 74.5% | `modal run modal_nla38.py::transfer_test` |
| 2 | Same 3.6 NLA, unchanged, on **Qwen3.8** | **0% extraction**: no `<explanation>` in any output | held-out 256 | row 1 | same |
| 3 | L42 activation drift 3.6 → 3.8 | centered cosine 0.82; FVE of 3.8 activations "predicted" by 3.6 activations = 61.9% | held-out 256 | — | same |
| 4 | Weight drift 3.6 → 3.8 | relative ‖ΔW‖/‖W‖: MLP 0.59–0.64, attention 0.33–0.52, embeddings 0.49, norms ≤ 0.08 | all tensors | — | `modal run modal_nla38.py::weight_diff` |
| 5 | AR SFT on gold explanations (1 epoch, 120k rows) | **58.4% FVE** | val 1,024 | identity-head init: −118% | `train_ar` (2×B200) |
| 6 | AV SFT (1 epoch, 120k rows) | val CE **1.192** vs **1.780** with shuffled activations | val 512 | step 0: 2.898 | `train_av` (4→2×B200, resumed at step 800) |
| 7 | Full SFT NLA (AV → AR) | **53.2% FVE**, 99.6% extraction | held-out 256 | shuffled-activation control: **−76.3%** | `modal run modal_nla38.py::evaluate --tag eval_sft` |
| 8 | Early SFT snapshot (AV ≈ step 600), standalone `nla_qwen38.py` | 54.3% FVE, 100% extraction | held-out first 64 | — | `modal run modal_nla38.py::test_infer` |
| 9 | GRPO, steps 0 → 10 → 20 → 30 (8×B200, 128 prompts × 8 rollouts/step), paused at step 30 | val FVE **52.6% → 61.8% → 64.2% → 65.8%**, 98–100% extraction | val 256 (buckets 5–25) | step 0 = SFT | `python spawn.py train_rl '{"max_usd": 50}'` |

Row 9 is scored by the AR critic, which is co-trained during RL (the standard NLA metric). The held-out-256 eval and the eval with the frozen SFT critic run when RL finishes.

The shuffled-activation control gives the AV the activation of a different row, and the AR scores
the result against the true activation. The FVE falls far below 0, so the explanations carry
row-specific information from the injected vector. They are not generic text.
