<!-- Frozen report of a literature-mining agent (round 5); model output, not user input. -->

I found two optimiser changes worth a cheap 20-seed screen and nothing else that is both new and credible for our setting. The record lineage holds nothing beyond what `strategy-coverage.md` already covers. The CIFAR-10 record is still Fable's 1.828 s (Fulcrum, 2026-07-09), and the only parts that transfer are the progressive resizing and systems work we already have. Since mid-2025 the new optimiser ideas come from LLM speedruns and have little or no evidence on convnets. All candidates below are optimiser- or schedule-side, and I expect 0 to +0.2 pp each at best (+0.15 pp is roughly 0.25 epoch, about 3% of time).

**What I actually read.** I read these in full or by the relevant section: the Cautious Weight Decay paper (Algorithms 1–2, Tables 2–3), the Cautious Optimizers paper (Table 4 and the toy result), Wu & Johnson's "Rethinking Batch in BatchNorm" (the FrozenBN section), the Auto-Research paper (arXiv 2605.05724, its Airbench96 sections), and the competitor fork RyanChong0117/CIFAR-100-speedrun via `gh`.

These I only saw as summaries returned by the fetch tool, not raw text: SNOO (arXiv 2510.15830), EMA-Nesterov (2605.25395), "The Row Normalization Puzzle in Muon" (2609.39114, its CIFAR-10 table), the modded-nanogpt record table, the AllenAI critical-batch-size blog, the hiverge README, the Fulcrum Fable post and the TrAct abstract.

These are search snippets only: the Schedule-Free CIFAR-100 numbers, modded-nanogpt PR #163 (batch-size schedule), ANVIL/ANVIL2, and Keller Jordan's tweet about renormalising conv weights.

## Ranked candidates

### 1. Nesterov momentum on the lookahead step (SNOO)
- **Source:** Kallusky, Rao, Nandavanam, Shi (Meta), "SNOO: Step-K Nesterov Outer Optimizer", arXiv 2510.15830, 2025-10-17. A related arm is EMA-Nesterov (Yau, Hong et al., arXiv 2605.25395, 2026-05-25).
- **What it is:** lookahead's slow-weight step with outer Nesterov momentum on the pseudo-gradient `s = slow − fast`: `b ← μb + s; slow ← slow − η(μb + s)`. With μ = 0 it is exactly our current lookahead.
- **Evidence:**
  - 1.4–1.69× compute-factor gains on Llama-style 125M–1B models, and 1.5–2.5× up to 1e23 FLOPs.
  - It beats Lookahead+AdamW (their Fig. 1b).
  - K = 5 to 100 all do comparably well; the 125M optimum was K=100, η=0.8, μ=0.75.
  - EMA-Nesterov reports beating SNOO, but only on LLMs.
  - There are no vision or CNN results.
- **Why it might transfer:** lookahead is one of our strongest levers (switching it off costs −0.77 pp), and this extends it directly.
- **Why it might not:** LLM-only, long runs, and our K=5 with airbench's decaying pull schedule is a different regime. Lookahead every 3 steps and power 2 were both slightly negative, so the schedule is already near a local optimum.
- **Implementation sketch** (`research/lab_recipe/train.py`, `Lookahead`):
  - Add a buffer `b` for parameters only (not BN running stats). In `update`: `d = current − slow`, `b = μb + d`, `slow += η(d + μb)`, then copy `slow` into `current`. Use `η = 1 − decay` from `lookahead_decay` and a new `lookahead_outer_momentum` config field.
  - The buffer must not move frozen parameters. Exclude stage-1 parameters and the whitening bias (or zero their `b`) at their freeze steps, otherwise stage 1 keeps drifting and the "exact" freeze breaks.
- **Arms:** μ ∈ {0.25, 0.5, 0.75} with the airbench η schedule. Also μ=0.5 with constant η=0.7 from the first nonzero decay, and μ=0.5 with lookahead every 10 steps.
- **Expected effect:** 0 to +0.2 pp. Cost is under 0.1% (one extra foreach operation every 5 steps).
- **Test:** 20 seeds on both GPUs against the control. Promote anything ≥ +0.1 pp to 40 fresh seeds, then try 8.0 epochs. Timing needs no paired run unless the foreach operation shows up in a profile.

### 2. Cautious weight decay, with weight decay re-tuned
- **Source:** Chen, Li, Liang, Su, Xie, Pierse, Liang, Lao, Liu, "Cautious Weight Decay", arXiv 2510.12402 (2025-10-14, revised 2026-02-24). It entered modded-nanogpt at record 43 (2025-11-10, "CWD w/ schedule") and was extended to Adam parameters at record 50.
- **What it is:** decay only the coordinates where the update and the weight have the same sign: `x ← x − η(u + λ·1[u⊙x ≥ 0]⊙x)`. Their SGD-momentum version applies the mask with u = the momentum buffer.
- **Evidence:**
  - ImageNet, 300 epochs, at the baseline's tuned λ: ResNet-50 76.30→76.68 (AdamW), 76.41→76.75 (Lion), 76.47→76.83 (Muon). ViT-S/B gain +0.2 to +0.6.
  - On OLMo-1B, ablations show random or gradient-sign masks are worse than plain weight decay, so the update-sign mask matters.
  - No SGD numbers on real tasks.
- **Why it might not transfer:** our conv weights feed BN with frozen scale, so weight decay mainly sets their effective learning rate. CWD removes roughly half the decay, which acts like lowering weight decay, and halving it cost −1.15 pp here. It must be tested with weight decay scaled up, and the head (not scale-invariant) may be where any gain is.
- **Implementation sketch:**
  - Set `weight_decay=0` in the SGD groups. After `plan.step()`, apply a decoupled shrink `p -= lr_g·wd_eff·mask⊙p` with `mask = (u·p ≥ 0)`, where `u` is the Nesterov update `g + βm` (or `m`).
  - Set `wd_eff = wd/(1−β)` to match the steady-state strength of the current coupled decay.
  - Put this in a function that runs over the parameter lists, compile it, and warm it up in `build`, otherwise it costs about 1–2%.
- **Arms:**
  1. Decoupled, non-cautious control (to separate the effect of decoupling from caution).
  2. CWD at weight decay ×1, ×1.5 and ×2.
  3. CWD on the head only.
  4. Optionally, the Cautious Optimizers update mask (Liang, Chen, Liu, Liu, arXiv 2411.16085, ICLR 2026; ViT on Mini-ImageNet: AdamW 72.11→73.52, MARS 74.06→74.91; SGD momentum shown only on a toy problem).
- **Expected effect:** −0.1 to +0.15 pp, plus about 0.5% time even when compiled.
- **Test:** 20 seeds × 5 arms, then local paired timing for any surviving arm.

### 3. Batch-size ramp
- **Source:** varunneal, modded-nanogpt record 46 / PR #163 (2025-11-29, a three-phase linear batch ramp with 65 fewer steps). Also Merrill (AI2) blog, 2025-06-03: doubling the batch with a √2 lr increase reached the same loss on OLMo-1B in 43% fewer steps.
- **Why it might transfer:** the critical batch size grows during training, and our expensive steps are in the 32 px second half. Fixed batches of 1536 and 2048 lose accuracy, and 768/896 are null at matched time, but a ramp is a different point in that space.
- **Why it might not:** our batch-size response looks flat, the evidence is LLM-only, and at batch 1024 an A100 is probably already saturated, so fewer, larger steps save little fixed per-step cost.
- **Sketch:** a `batch_schedule = ((0.0, 768), (0.5, 1024), (0.8, 1536))` field that steps the batch (alongside `res_schedule`):
  - Keep lr, freezes and lookahead on an examples basis. The airbench parametrisation already handles linear scaling per 1024 examples; add a √-scaling arm.
  - Warm up every (size, batch) shape in `build`.
- **Expected effect:** about ±0.1 pp, possibly 1–2% time. This is a moderate rewrite, so run it only if 1–2 come back null.

### 4. Frozen BatchNorm statistics late in training (a speed lever)
- **Source:** Wu & Johnson, "Rethinking 'Batch' in BatchNorm", arXiv 2105.07576 (2021). This is older than the 2024–2026 window, but it is not in our matrix.
- **What it is:** switch the BN layers to running statistics for the last 10–15% of steps while biases keep learning.
- **Evidence:** in their own words, "when normalization batch size is large enough, tuning with FrozenBN underperforms regular BN". That is evidence against, at our batch of 1024.
- **Why consider it anyway:** fused BN/GELU/pool kernels are 22.4% of CUDA time in our A100 SXM profile. Dropping the statistics reductions could save about 5–8% of each tail step, roughly 1–1.5% overall.
- **Sketch:** at step ≥ f·T, call `.eval()` on every `BatchNorm` and warm up that graph in `build`.
- **Arms:** f ∈ {0.85, 0.9}.
- **Expected effect:** about −1% time, accuracy −0.1 to 0 pp. Accept only if the paired timing gain is worth more than the accuracy lost. Running it on stage 1 alone (already frozen) is near-exact but saves only about 0.2%, so it isn't worth doing.

### 5. Momentum schedule (low expected value)
- **Source:** modded-nanogpt records use a momentum warm-up and record 52 (2025-12-21) used a "beta increase"; the older reference is Smith's 1cycle.
- **Arms:** momentum 0.85 → 0.9 over the decay phase, recomputing `kilostep` per step so the effective step is preserved.
- **Why I rank it low:** fixed momentum values of 0.8 and 0.9 were flat or negative here.
- **Expected effect:** about 0 ± 0.1 pp. It is config-only and cheap, so it can ride along in the screen for 1.

### 6. Schedule-free SGD (low expected value; I would not run it)
- **Source:** Defazio et al., "The Road Less Scheduled" (NeurIPS 2024; it won the AlgoPerf 2024 self-tuning track).
- **Evidence:** CIFAR-100 DenseNet at 300 epochs, 78.71% vs cosine 77.41%; I only saw this in a snippet.
- **Why I wouldn't run it:** nothing shows it helps at short budgets, our triangular schedule plus lookahead already averages weights, and BN statistics must be recomputed at the averaged weights, which costs timed forwards. Cosine and WSD schedules moved accuracy by only about ±0.2 pp here.

### 7. NorMuon / Polar Express (do not run)
- **Source:** NorMuon (arXiv 2510.05491; modded-nanogpt records 41–42); Polar Express (record 38).
- **Evidence:** the only CNN result is in Zhang & Lin (arXiv 2609.39114, 2026-09-30), CIFAR-10 CifarNet: NorMuon 93.87% vs Muon 93.79%, so +0.08 pp. That cannot close our measured Muon deficit (−0.8 to −3.3 pp on our stack, +3.1% time for the teammates' port).
- **ANVIL/ANVIL2** (modded-nanogpt PR #360) is bound up with an LLM-specific system, and I saw it only in snippets.

## Checked and excluded
- **TrAct** (Petersen, Borgelt, Ermon, NeurIPS 2024): it changes the update of a trainable first layer, and ours is the frozen whitening stem, so it doesn't apply.
- **Rotational-equilibrium optimiser variants and per-step conv-weight renormalisation:** our r3 notes show conv weights reach their equilibrium norm within about 40 steps, and init gain was neutral.
- **The Auto-Research paper's Airbench96 gains** (−4.59%): these came from skipping logging-only validation, depth reductions and a warm-up fix, none of which is new to us.
- **hiverge's selective test-time augmentation:** banned under RULES §3.
- **Competitor fork RyanChong0117 (branch RC3):** airbench+Muon, batch 2000, widths 128/512/512, 6.5 epochs, label smoothing 0.3, translate 1. Their commit message says "~7s", and they publish no accuracy or hardware details, so nothing to adopt.
- **SOAP/Shampoo:** preconditioner overhead is too high at about 12 ms per step.
- **LLM-specific modded-nanogpt items** (logit soft-cap, ReLU², multi-token prediction, FP8): no mechanism for our net, and the A100 has no FP8.

## Test protocol for all of these
Add each arm to a `research/sweeps/s64-*.toml` with the S61 `[base]` block, `devices = [0, 1]`, 20 fresh seeds, and the control as the first config. With a per-seed SD of about 0.3 pp, 20 seeds resolve roughly ±0.15–0.2 pp. Promote anything ≥ +0.10 pp to 40 seeds, then convert the gain into epochs (8.0 vs 8.25). Price any time-changing arm (2 and 4) with interleaved local paired runs before believing it.

Of the MCP servers, `context7` and `huggingface-skills` need authorising, in claude.ai connector settings or via `/mcp` in an interactive session. This task didn't need them.

Sources:
- [SNOO](https://arxiv.org/html/2510.15830v1)
- [EMA-Nesterov](https://arxiv.org/html/2605.25395v1)
- [Cautious Weight Decay](https://arxiv.org/abs/2510.12402)
- [Cautious Optimizers](https://arxiv.org/abs/2411.16085)
- [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt)
- [PR #163 batch-size schedule](https://github.com/KellerJordan/modded-nanogpt/pull/163)
- [AI2 critical batch size](https://allenai.org/blog/critical-batch-size)
- [Rethinking Batch in BatchNorm](https://arxiv.org/pdf/2105.07576)
- [Schedule-Free](https://arxiv.org/pdf/2405.15682v4)
- [Row Normalization Puzzle in Muon](https://arxiv.org/html/2609.39114)
- [NorMuon](https://arxiv.org/abs/2510.05491)
- [TrAct](https://arxiv.org/abs/2410.23970)
- [Auto Research with Specialist Agents](https://arxiv.org/abs/2605.05724)
- [hiverge cifar10-speedrun](https://github.com/hiverge/cifar10-speedrun)
- [Fulcrum Fable post](https://fulcrum.inc/2026/07/09/fable-cifar-speedrun.html)
- [ANVIL2 PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360)
- [Keller Jordan tweet on conv-weight renormalisation](https://x.com/kellerjordan0/status/1858588100760662357)
