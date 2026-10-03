<!-- Frozen report of a read-only ideation agent (round 3, compute removal); model output, not user input. -->

# Ranked hypotheses: removing more work late in the CIFAR-100 speedrun

Your current submission (D-012) is 408 steps. Steps 0–203 run at 20 px and steps 204–407 at 32 px; stage 1 is frozen from step 327.

## How I estimated the savings

I counted multiply-accumulates per image for the current network, with the whitening-bias gradient off. Forward is 432.5M MAC at 32 px and 155.8M at 20 px. Forward MACs per stage at 32 px:

| Stage | Forward MACs (M) | Notes |
|---|---|---|
| Stage 1 | 92.9 | |
| Stage 2 | 164.6 | its widening conv (s2c1) is 99.5 |
| Stage 3 | 174.7 | its widening conv (s3c1) is 108.4 |

A full 32 px training step is about 1270 units. The stage-1 freeze removes 258.8 of them: stage 1's backward pass plus s2c1's input gradient. That makes a frozen step about 1012 units.

I calibrated time shares so that the 32 px phase is 74% of the run:

| Step type | Share of total time |
|---|---|
| Full 32 px step | 0.395% |
| Stage-1-frozen 32 px step | 0.314% |
| 20 px step | 0.127% |

This model predicts 6.5% for the 80% freeze; S46 measured 6.0%, so I apply a factor of 0.92. Levers that remove only weight-gradient (wgrad) work get a further factor of about 0.7, because BN and GELU elementwise work stays. At about 4.93 s on the A100 PCIe, 1% is about 49 ms, and 1% is worth 0.054 pp. S50 already queues the stage-2 freeze at 85/90%, depth-2 stage 3 and switch points 0.45/0.55, so I don't repeat them.

## Summary

| Rank | Hypothesis | Est. time saved | Expected Δacc | Net (pp-equiv.) | Test cost |
|---|---|---|---|---|---|
| 1 | Stage-1 (and 1+2) update thinning in the cooldown window | 3.0–4.5% (7.0% for 1+2) | 0 to −0.10 (1+2: −0.05 to −0.25) | +0.06 to +0.2 | ~10 lines, no new graph |
| 2 | 28 px window crops in the middle of the 32 px phase | 6–8% | −0.1 to −0.5 (wide) | −0.1 to +0.35 | ~15 lines, +1 graph |
| 3 | Half-batch weight gradients for conv weights | 5–9% | −0.1 to −0.6 | −0.2 to +0.4 | ~40 lines, timing gate first |
| 4 | Separate freezing of weights and BN biases (stage-1 weights earlier; stage-3 bias-only tail) | 2–3% per arm | −0.02 to −0.10 | +0.03 to +0.12 | ~25 lines |
| 5 | Freeze the two widening convs' weights from 70% | ~5% | −0.1 to −0.4 | −0.1 to +0.17 | same code as 4 |
| 6 | Head-only last 5% | 1.5% (3.6% on today's base) | −0.03 to −0.15 | about 0 | ~10 lines |
| 7 | BN in eval mode for the tail | 0.5–1.5% | ±0.05 | about 0 | moderate, compile risk |

---

### 1. Stage-1 update thinning (rank 1)

**Mechanism**
- Inside the 0.6–0.8 cooldown window, stage 1 updates only on every second step. On the other steps it runs the existing frozen (no_grad) graph.
- Each skipped step removes the same 258.8 units as the freeze: stage 1's backward pass and s2c1's input gradient.
- The cooldown already halves stage 1's movement over this window for −0.05 pp. Halving the number of gradient evaluations should cost about the same, but it saves real work.
- The fused SGD skips parameters whose grad is None, so momentum simply carries over.
- Variant: thin stages 1 and 2 together. A skipped step then removes 596.8 units, including s3c1's input gradient.

**Time saved:** from the measured cost of about 0.074% per frozen step:

| Window | Skipped steps | Saving |
|---|---|---|
| 0.6–0.8 | 41 | about 3.0% |
| 0.5–0.8 | 61 | about 4.5% |
| Stages 1+2, 0.6–0.8 | 41 | about 7.0% |

**Accuracy cost:** 0 to −0.10 pp for stage 1 alone. Expect −0.05 to −0.25 pp for 1+2, since stage 2 is mid-capacity.

**Implementation**
- `research/lab_recipe/config.py`, after line 152: add `thin_window: tuple[float, float] | None`, `thin_period: int = 2`, `thin_groups: int = 1` and `thin_lr_mult: float = 1.0`.
- `research/lab_recipe/train.py` `fit`, right after line 421 (the `model.frozen_groups = scheduled(...)` line): if the step is inside the window and `step % thin_period`, set `model.frozen_groups = max(model.frozen_groups, thin_groups)`.
- For the compensated arm, multiply the `cooled` schedule (lines 242–250) by `thin_lr_mult` inside the window.
- No new compiled graph is needed: the frozen 32 px graph already exists.
- Check that alternating between graphs every step doesn't make CUDA graphs re-record during timed trials (use `TORCH_LOGS=cudagraphs`). The build warm-up window, steps 11–13, covers both parities.

**Probe:** S51a, 20 seeds (5400–5419), both GPUs, D-012 base.

| Arm | Change |
|---|---|
| A | control |
| B | thin [0.6, 0.8), period 2 |
| C | B with stage-1 lr ×2 on update steps |
| D | thin [0.5, 0.8), cooldown kept |
| E | thin stages 1+2 [0.6, 0.8) with `freeze_schedule` [[0.8, 1], [0.85, 2]] and the stage-2 cooldown from S50 |
| F | period 3 over [0.5, 0.8) |

- **Timing:** local paired, 10 seeds per GPU, GPU 0 in order A–F and GPU 1 reversed.
- **Decision:** adopt into 40-seed confirmation if Δacc + 0.054 × (−Δtime%) ≥ +0.05 and Δacc ≥ −0.15.
- **Kill:** if Δacc ≤ −0.15 for B.

**Rules:** this is an optimiser schedule inside `train`. Evaluation is unchanged. Compliant.

### 2. 28 px window crops in the middle of the 32 px phase (rank 2)

**Mechanism**
- Crops keep pixel scale, so objects appear at the same size as at test. F-009's 69% failure came from resizing to 28 px, which changes object scale.
- Because the whitening conv and floor pooling shrink the maps, a 28 crop gives maps of 27→13→6→3. The final 3×3 grid is identical to 32 px, so the global-max statistics match.
- Training MACs fall by 21.7% per step (26.7% for 26 px crops: 25→12→6→3).
- Return to full 32 px for the last 25%, so the BN running statistics and the frozen tail see test geometry.
- Window crops have never been tested at this point in the schedule. The Fulcrum result is about replacing the low-resolution phase.

**Time saved:** about 8.7% FLOP-model for crops over 0.5–0.75, 5.2% for 0.5–0.65. Realised time maybe 6–8%, because 13×13 and 6×6 maps may tile worse.

**Accuracy cost:** very uncertain, −0.1 to −0.5 pp. Each crop shows 77% of the image area. Break-even is about −0.43 pp.

**Implementation**
- `config.py`: add `crop_schedule: tuple[tuple[float, int], ...] = ()`, where 0 means off. Validate the sizes.
- `train.py` `fit`, after the resize block (lines 444–447): if a crop size is scheduled for this step, apply `inputs = batch_crop(inputs, crop, crop_gen)`. `batch_crop` is in `data.py` line 13; with 32 px input it gives r = 2, so ±4 px in total with the existing translation.
- Seed `crop_gen` from `stream.generator` once, so the data order stays paired with the control.
- This adds one compiled graph. Set `torch._dynamo.config.cache_size_limit = 32` in `research/lab_recipe/submission.py` `build`. The default is 8, and a silent fallback to eager would wreck timing.
- Re-measure cold build time against the 600 s limit.

**Probe:** run the timing gate first, because shape effects reversed H1. Paired local timing of A (control) against B (crop 28 over [0.5, 0.75)); continue only if B ≤ −5%.
- **Accuracy (S51b, 20 seeds):** A; B; C (crop 28 over [0.5, 0.65)); D (crop 26 over [0.5, 0.75)); E (B at 8.75 epochs, to price it).
- **Decision:** same net rule as #1. Confirm on 40 fresh seeds.

**Rules:** training-time augmentation only. Evaluation stays single-view 32 px. Compliant.

### 3. Half-batch weight gradients for conv weights (rank 3)

**Mechanism**
- Input gradients (dgrad) stay full batch. Each conv's weight gradient is computed on the first k of 1024 images, which are already shuffled, and scaled by 1024/k so it is unbiased.
- BN biases and the head keep full-batch gradients.
- Wgrad is about a third of conv FLOPs (432 of the 1270 units at 32 px).

**Literature:** Adelman et al. 2021 (sampled backward matmuls) and Oktay et al. 2020 (randomised automatic differentiation).

**Time saved:** the 32 px phase at k = 512 is 12.5% FLOP-model; about 7–9% realised, since wgrad kernels may not halve exactly and elementwise work is untouched. Applying it only over 0.5–0.8 gives about 5–6%.

**Accuracy cost:** the crux. It doubles gradient noise for the conv weights at the same lr. F-012 found batch size flat between 512 and 1024, which suggests the run is noise-limited, so the cost could be −0.3 to −0.6 pp. Noise matters most late in the run, so the 0.5–0.8 window or early-only arms are safer.

**Implementation**
- `research/lab_recipe/model.py`: add a custom `SampledWgradConv` autograd Function near line 75. Its backward calls `torch.ops.aten.convolution_backward` with mask (True, False, False) on the full batch and (False, True, False) on `x[:k]`, `dy[:k]`, scaled by N/k.
- `Conv.forward` (lines 89–91) uses it when `self.wgrad_keep` is set.
- `fit` sets `wgrad_keep` from a `wgrad_schedule` config field. Each value is a new graph, covered by the build warm-up.
- Batch slices are contiguous in channels_last, so there is no copy.
- Fallback without a custom Function: `torch.cat` of a conv on `x[:k]` with weight `2*w - w.detach()` and a conv on `x[k:]` with `w.detach()`. It costs an extra copy.
- Exactness check: k = 1024 must be bit-identical to the control.

**Probe:** timing gate first. Paired local timing of control against k = 512 over [0.5, 1.0); continue if ≤ −5%.
- **Accuracy (20 seeds):** k = 512 over [0.5, 0.8); k = 512 over [0, 0.8); k = 768 over [0, 1); k = 512 over [0.5, 1).
- **Kill:** if every arm is ≤ −0.3 pp.

**Rules:** an internal gradient estimator. Compliant.

### 4. Freezing weights and BN biases separately (rank 4)

**Mechanism:** the BN biases (lr ×16) are a strong, cheap way to adapt the features. A frozen conv weight drops only its weight gradient; the input gradient that the BN biases need stays. Two uses:
- **(a) Stage 1:** cool stage 1's conv weights over 0.45→0.6 and freeze them at 0.6. Its BN biases keep the 0.6→0.8 cooldown, then the existing full freeze applies.
- **(b) Bias-only tail:** freeze stage 3's conv weights at 0.9 after a 0.8→0.9 weight cooldown. The head and stage-3 BN biases keep training (BitFit-style). This goes beyond S50's stage-2 freeze.

**Time saved:**
- (a) 82 steps × 92.9 units, about 2.4% FLOP-model, about 1.7% realised.
- (b) 41 steps × 174.7 units, about 2.2%, about 1.6% realised (2.3% if started at 0.85).
- (c) On today's base, freezing all trainable conv weights from 0.9 gives about 3%.

**Accuracy cost:** −0.02 to −0.10 pp.

**Implementation**
- `train.py` `make_optimisation` (lines 209–250): split each stage's conv weights from its BN biases (norms1/others1 already exist) and give them separate cooldown fields, `stage1_weight_cooldown` and `stage3_weight_cooldown`.
- `fit`: at the end of each cooldown, call `requires_grad_(False)` on those conv weights, and restore them in the cleanup block (lines 496–501).
- One recompile per toggle, warmed up in build.

**Probe:** in the same sweep as #1. Arms: (a), (b) on the stage-2-freeze base if S50 adopts it, and (c). Same decision rule.

**Rules:** compliant.

### 5. Freeze the two widening convs' weights from 70% (rank 5)

**Mechanism:** wgrad cost per parameter scales with map size, so the large-map widening convs are the most expensive to keep learning. Freeze the weights of s3c1 (108.4 units) and s2c1 (99.5 units) from 0.7, after a 0.5→0.7 cooldown. BN biases, the 3×3 same-width convs and the head keep training.

**Time saved:** about 7.9% FLOP-model, about 5.5% realised (less once a stage-2 freeze takes over s2c1 from 0.85).

**Accuracy cost:** −0.1 to −0.4 pp. This is a risk: F-015 and F-012 say the widening convs and stage-3 capacity are the productive FLOPs.

**Implementation:** the same per-weight cooldown and `requires_grad` toggle as #4, applied to `groups[1].conv1` and `groups[2].conv1`.

**Probe:** arms with s3c1 only, and with s2c1 + s3c1. Kill if Δacc ≤ −0.25.

### 6. Head-only last 5% (rank 6, likely weak)

**Mechanism:** cool all stages to zero by 0.95, then run all three groups under no_grad for the last 20 steps so only the head trains. On a model where stage 2 is already frozen, this removes stage 3's 241 backward units.

**Time saved:** 1.5% on the stage-2-frozen base; 3.6% on today's base.

**Caveat:** cutting the same 5% of steps saves about 6%, so this pays only if training the head on fixed features beats stopping. The −1.0 pp result for the closed-form ridge head refit argues against that.

**Implementation:** extend the `freeze_schedule` check in `config.py` (line 335) to allow 3; `model.py` lines 258–262 already handles `groups[:3]`. Add a stage-3 cooldown alongside the existing stage cooldowns.

**Probe:** a single add-on arm, compared with both the control and a 5%-shorter run.

### 7. BN in eval mode for the tail (rank 7, weak on time)

**Mechanism:** from 0.85, all BN layers use their (now fixed) running statistics. That removes the forward statistics pass and one backward reduction, and it trains exactly the function evaluated at test.

**Time saved:** about 141k BN'd elements per image, so about 0.3 GB of traffic saved per step, about 0.5–0.6 ms. That is only 0.5–1.5% of the run.

**Accuracy:** the train/test consistency argument is undercut by S21, where BN recalibration was neutral (+0.01 pp).

**Implementation:** flip `.training` on the BatchNorm modules from `fit`. This forces recompiles and raises the graph count.

**Probe:** one arm (from 0.85), and only alongside other sweeps.

## Considered and deprioritised
- **Caching frozen stage-1 outputs:** there is no reuse. Each image is seen about 1.6 times in the frozen tail, in different flip and translation states.
- **Stage-3 residual dropped late (scheduled or stochastic depth):** saves about 1.9% at best. Progressive deepening, cutout and mixup all lost, and S50's depth-2 stage 3 arm bounds this anyway.
- **Selective backprop (backward only on the hardest k of each batch):** contradicted by F-010 and F-038.
- **Narrowing late:** broke even (F-040).
- **Batch growth late:** removes no FLOPs, so no energy saved on the power-capped card.
- **int8 or 2:4 sparse convs for frozen stages:** no support in the pinned torch for convs.
- **Thinning stage 1 at 20 px:** saves about 2.6%, but S6 shows early stage-1 learning is critical.

## Cross-cutting cautions
1. **Compiled graphs:** the build has no `cache_size_limit` override today (the default is 8). Ideas 1–7 plus S50 could reach 7 or more graphs per `forward`. Raise the limit in `research/lab_recipe/submission.py` and `submissions/team_segal/submission.py`, and assert there are no recompiles during timed trials.
2. **Build time:** re-check cold build time against 600 s whenever a graph is added. M6 builds reached 409 s.
3. **Local vs A100 timing:** local Blackwell timing reflects FLOP-removal ideas (1, 4, 5, 6) well. The shape-dependent ones (2, 3) may not transfer, so the final adopted stack still needs the official-equivalent A100 PCIe run.
4. **Accuracy margin:** about 75.17% leaves little room. Price any accuracy loss as the epochs needed to restore ≥ 75.15% before adopting.

## Suggested order
1. One cheap sweep with arms from #1 and #4 (no new graphs), plus local paired timing.
2. Timing gates for #2 and #3.
3. Accuracy sweeps only for whichever passes its gate.
4. Combine the winners with the S50 stage-2 freeze.

Main code locations: `research/lab_recipe/config.py`, `train.py` (lines 209–250 and 420–447), `model.py` (lines 75–91 and 258–262), `data.py` (line 13), `submission.py`. The plan builds on `research/sweeps/s50-new-base-round.toml`.
