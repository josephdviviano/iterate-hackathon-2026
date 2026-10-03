<!-- Frozen report of a read-only ideation agent (round 3, learning per step); model output, not user input. -->

# CIFAR-100 speedrun: ranked hypotheses for more learning per step

I generated 9 new hypotheses and ranked them by expected gain per GPU-minute. The strongest are new uses of the two levers that already worked, the BN-bias update and the structured init, where the existing sweeps leave a mechanism question open. Nothing in the tables shows a lever worth more than about +0.15 pp. Expect at most one or two of these to survive, which would buy roughly 0.25 epoch. This was read-only: I changed no files and ran no jobs.

**Check S50 before running anything.** S50 is running now on the D-012 base (`results/sweeps/s50-new-base-round`). It already covers BN-bias scaler 8 and 32, lr 10 and 13, warm-up peak 0.15, switch at 0.45 and 0.55, depth-2 stage 3, and a stage-2 freeze. None of the hypotheses below repeats those. H1 and H2 should take whatever scaler S50 picks as their base.

**Shared protocol:**
- All runs are local on GPUs 0 and 1, never Modal, with an in-sweep control and the same 20 seeds for every arm.
- The per-arm SE is about 0.05 pp. With about 10 arms the best-of-k noise ceiling is about +0.12 to +0.15 pp.
- Promote an arm at +0.10 pp or more, and only when the pattern across arms fits the stated mechanism. Then confirm on 40 fresh seeds.
- A confirmed gain of 0.08 pp or more earns an epoch-cut confirmation (8.5 → 8.25) on 40 more fresh seeds.
- I estimate a 20-seed arm costs about 4 to 5 GPU-minutes, from the per-trial times in the tables plus the compile.
- H1, H4, H5 and H7 can share one control in a single sweep of about 12 arms, roughly one hour on two GPUs.

One fact from the code underlies several items. BN-bias decay is decoupled from the scaler: the group sets `weight_decay = wd/lr_bias`. That gives a per-step shrink of about wd/(1−μ) ≈ 0.0133 × schedule, a horizon of roughly 75 steps at peak, and about e⁻³ of memory over the 408 steps. So the scaler sets both the noise in the bias updates and their equilibrium size, which is roughly proportional to lr_b/wd_b. Every scaler sweep so far changed both together.

## Ranked hypotheses

### H1. Separate bias size from bias noise: a bias weight-decay multiplier
*Rank 1: zero time cost, about 25 GPU-min, and it explains the strongest lever.*

- **Mechanism.** Moving the scaler from 64 to 16 cut the bias step 4× and, through the decoupled decay, also cut the equilibrium bias size 4×. Two explanations fit:
  - **Size:** biases should be small, which keeps the frozen-scale GELUs near their linear operating point.
  - **Noise:** bias updates should be slow and quiet.
  - In an OU picture the equilibrium size goes as scaler/mult, the noise SD as scaler/√mult, and the forgetting rate as mult.
- **Evidence.** The scaler curve is inverted-U (64: 0, 32: +0.19, 16: +0.30/+0.34, 8: +0.23, 4: +0.07, 2: −0.25 pp; F-039, F-043). F-036 tested conv and head weight decay but never bias weight decay.
- **Implementation.** Add `bias_wd_mult: float = 1.0` to `research/lab_recipe/config.py`. In `make_optimisation` (`train.py`), set the norm, norm1 and norm2 groups to `weight_decay = wd * bias_wd_mult / lr_bias`.
- **Probe (20 seeds).**

  | Arm | Scaler | Mult | Prediction |
  |---|---|---|---|
  | Control | 16 | 1 | — |
  | R1 | 16 | 0.25 | Size like scaler 64, less noise |
  | R2 | 64 | 4 | Size like scaler 16, more noise |
  | R3 | 16 | 4 | — |
  | R4 | 16 | 0 | — |

  - Size mechanism: R2 ≈ control and R1 ≈ −0.3 pp.
  - Noise mechanism: R1 ≈ control and R2 ≈ −0.3 pp.
- **Expected.** Mainly a diagnostic. If size is the mechanism, R3 or a mult of 2 may add +0.05 to +0.15 pp, a new axis independent of the scaler.
- **Kill rule.** Stop the axis if R1 to R4 all sit within ±0.08 pp of control and the R1/R2 pattern points to noise.
- **Rules.** Scalar hyperparameters only.

### H2. Per-stage BN-bias learning rate set from measured gradient scale
*Rank 2: a 2-seed diagnostic first, then about 15 GPU-min.*

- **Mechanism.** A bias gradient sums dL/dy over N·H·W positions:
  - At 32 px that is 15² positions in stage 1, 7² in stage 2 and 3² in stage 3.
  - At 20 px it is 9², 4² and 2².
  - So the update size per bias, in units of pre-activation SD, likely differs several-fold between stages. It also jumps 2.3 to 3× at the switch.
  - One global scaler can only be a compromise.
- **Evidence.**
  - The optimum is flat over a 4× band (8 to 32 gives +0.19 to +0.34 pp), so per-stage tuning helps only if the stages differ by more than about 2×.
  - The phase-switched scaler did not replicate (F-043), which is why this is per-stage rather than per-phase.
  - S39/S41 suggest most of the gain arises in the 20 px phase.
- **Implementation.**
  - Add `bias_scaler_stages: tuple[float, float, float] | None`, multipliers on `lr_bias`.
  - Split `norms` by `model.groups[i]` membership, as the `stage1_cooldown` code already does, and apply the stage-1 multiplier to `norms1`.
  - Diagnostic flag: log the per-stage RMS of lr × grad for BN biases every 10 steps.
- **Probe.**
  - Diagnostic: 2 seeds of control, logging the per-stage RMS.
  - Arms: (i) multipliers that equalise the per-stage update RMS, keeping the geometric mean at the base; (ii) the inverse, as a mechanism check; (iii) control.
- **Expected.** +0.0 to +0.15 pp.
- **Kill rule.** Kill without arms if the per-stage RMS differs by less than 1.5× at both resolutions. Reject if arm (i) does not beat arm (ii) by 0.1 pp or more.
- **Rules.** Scalars only.

### H3. Steep, later stage-1 cooldown to allow an earlier freeze
*Rank 3: a direct time lever, about 40 GPU-min with paired timing.*

- **Mechanism.** In S44 the cooldown from 0.5 to 0.7 lost 0.20 pp against 0.6 to 0.8. That window starts exactly at the 20→32 px switch, so stage 1 never adapts at full lr to the new input statistics. The loss may come from that timing, not from total stage-1 learning. A steep cooldown leaves full lr through the first part of the 32 px phase, then drops quickly.
- **Evidence and cost.**
  - Freeze at 80% gave −6.0% time, at 75% −7.7% (S46). About 1.7% per 5% of training, so 70% should give about −9.4%.
  - That is about 3.4% better than now, worth about 0.18 pp at the exchange rate.
  - Freeze at 75% already costs only −0.06 pp against 80% (S45), which is roughly break-even.
- **Implementation.** Existing fields `stage1_cooldown` and `freeze_schedule`. The boost arm needs a new `stage1_lr_mult: float = 1.0` on the `norms1`/`others1` base lr.
- **Probe (20 seeds).**
  - Control: (0.6, 0.8), freeze 0.8.
  - (0.65, 0.75), freeze 0.75.
  - (0.6, 0.7), freeze 0.7.
  - (0.6, 0.7), freeze 0.7, with `stage1_lr_mult` 1.3.
  - Winners get 40 fresh seeds plus S46-style paired local timing: GPU 0 in order, GPU 1 reversed, 10 seeds each, max-autotune.
- **Expected.** −1.5 to −3% time at equal accuracy.
- **Kill rule.** If (0.6, 0.7) is no better than S44's (0.5, 0.7) by at least 0.05 pp, the cost is total stage-1 learning, not timing. Stop.
- **Rules.** Exact freeze, no new state.

### H4. Identity plus a structured perturbation for the square convs
*Rank 4: zero time cost, about 25 GPU-min.*

- **Mechanism.** These convs start as exact identity, so they add no spatial filtering at step 0:
  - stage-1 conv2 and residual;
  - stage-2 conv2;
  - stage-3 conv2 and residual;
  - and the first cin rows of each widening conv.
  
  `W = dirac + β·D` keeps the identity path, which SkipInit showed the short run needs (gates at 0 or 0.25 cost 0.6 to 1.0 pp, F-046). D is a DCT bank with orthonormal channel mixing at the Kaiming row norm. It starts these layers with frequency-selective filtering, as the widening rows already do.
- **Evidence.** DCT on the widening rows gave +0.11 to +0.17 pp. Orthogonal and ZerO inits were neutral, so the gain came from frequency structure, not conditioning.
- **Implementation.**
  - Add `square_init: str = "dirac"`, one of `"dirac"`, `"dirac+dct"` or `"dirac+kaiming"`, plus `square_beta: float`.
  - In `model.py` `reset_model`, for `Conv` with `cout == cin`, add β times the `structured_init_("dct")` construction over all rows.
  - Add `widen_identity_beta` to apply the same to the identity rows of the widening convs.
- **Probe (20 seeds).** dct β 0.25; dct β 0.5; Kaiming β 0.5, which separates structure from plain symmetry-breaking; widening identity rows with dct β 0.5; control.
- **Expected.** +0.0 to +0.12 pp.
- **Kill rule.** Reject if no DCT arm beats the Kaiming arm and control by 0.08 pp or more.
- **Rules.** A mathematical basis plus per-trial random mixing from the seeded generator, as in the adopted DCT init.

### H5. Centre the pooled features before the head
*Rank 5: about 0% time, about 15 GPU-min, high variance.*

- **Mechanism.** Flatten-max of `x + GELU(·)` gives strictly positive features with a large mean.
  - The head's curvature E[ffᵀ] = Σ + μμᵀ then has one eigenvalue about d·(μ/σ)² larger than the rest.
  - That common-mode direction acts as an implicit per-class bias and caps the usable head lr.
  - This is LeCun's centred-inputs argument applied to the head.
- **Evidence.** Head lr ×2 cost −0.39 pp and ×0.5 cost −0.18 pp (F-036): a sharp optimum, consistent with a curvature limit. Centring has never been tested; F-036's changes to head geometry (cosine, ETF, class-mean init, expansion) were all different.
- **Implementation.**
  - Add `head_center: bool`.
  - In `AirbenchNet._head`, after `global_max`, apply an fp32 `BatchNorm1d(w3, affine=False)`, or a mean-only variant using a running-mean buffer with BN momentum 0.6.
  - `reset_model` already resets its buffers. Cast back to fp16 before the head.
- **Probe (20 seeds).** Centre; centre with `head_lr_mult` 2; control.
- **Expected.** −0.1 to +0.2 pp.
- **Kill rule.** Reject if both arms are at or below control. If centring alone is about 0 but lr ×2 no longer hurts, the mechanism is confirmed but there is no gain; stop.
- **Rules.** Standard BN eval-mode running statistics, accumulated during timed training. No adaptation at evaluation.

### H6. Whitening fitted at 32 px but used on 20 px inputs for half the run
*Rank 6: CPU-only diagnostic first.*

- **Mechanism.**
  - The 2×2 whitening is eigendecomposed on 32 px patches, but the first half trains on antialiased 20 px images.
  - Those have lower neighbour correlation, so the small-eigenvalue (high-frequency) channels are over-amplified in the 20 px phase, then drop at the switch.
  - eps 5e-4 is small next to the smallest eigenvalues, which amplifies noise further.
  - Neither eps nor the resolution of the whitening statistics has been tested.
- **Implementation.**
  - Add `whiten_eps` and `whiten_mix_res: int = 0`.
  - With `whiten_mix_res` set to 20, `init_whitening` pools covariance from both the images and their 20 px downsample, at about 1 ms of timed `prepare`.
- **Probe.**
  - Diagnostic, CPU only: per-channel variance of the whitening output at 20 px versus 32 px on 5,000 images.
  - Arms: eps 5e-3; eps 5e-2; mixed-resolution covariance; control.
- **Expected.** 0 to +0.1 pp.
- **Kill rule.** Skip the mixed-resolution arm if no channel's variance ratio exceeds 1.5×. Reject eps if both arms are at or below control.
- **Rules.** Data-derived init inside timed `prepare`, as now.

### H7. BN-bias init: a mean shift or threshold diversity
*Rank 7: weak.*

- **Mechanism.** All channels start at the same GELU threshold. Fixed, spread thresholds, as with random-feature biases, could diversify early units, and a lower scaler makes the init last longer.
- **Against it.** The roughly 75-step decay horizon erases any init by about step 150.
- **Implementation.** Add `bn_bias_init_spread` and `bn_bias_init_mean`. In `reset_model`, set each BN bias to `mean + a·(2(randperm(C)+0.5)/C − 1)`.
- **Probe.** First inspect the trained biases of 2 control models (per-layer mean, SD, fraction below −1). If they cluster at a consistent mean, run that mean as a scalar; otherwise run spread 0.5. Control alongside.
- **Expected.** −0.1 to +0.08 pp.
- **Kill rule.** Skip the arms if the trained SD is below 0.2 and the mean is near 0.
- **Rules.** Scalars only. Per-channel vectors copied from trained runs would be learned state and are banned.

### H8. Trainable BN gain with its own small lr
*Rank 8: weak.*

- **Mechanism.** A per-channel γ adds a learnable residual mix and GELU regime per channel while the convs stay scale-invariant.
- **Implementation.** Add `bn_scale_train` and `bn_scale_scaler` (1 or 4), with no weight decay on γ, since decay would pull it toward 0.
- **Expected.** −0.15 to +0.1 pp. airbench froze γ deliberately.
- **Kill rule.** Reject if at or below control.

### H9. Translation strength by resolution phase
*Rank 9: weak.*

- **Mechanism.** The crop happens at 32 px before the downsample, so the 20 px phase effectively sees only about 1.25 px of shift.
- **Implementation.** Add `translate_phases: tuple[int, int]`: pad to the maximum, centre-crop to the phase radius, then `batch_crop`.
- **Probe.** (3, 2); (1, 2); control.
- **Evidence against.** Translate 1 and 4 over the whole run were both rejected, and the clean tail was null.
- **Expected.** About 0 ± 0.1 pp.

## Considered and not worth GPU time
- **Label smoothing at the new budget:** levels 0.1 to 0.3 are flat, annealing cost −0.21 pp and online LS −0.10 pp.
- **Momentum or lookahead changes around the freeze:** once stage 1's lr is 0, the next lookahead sets current equal to slow and nothing moves after that. There is no mechanism to exploit.
- **Zero or orthogonal head init:** initial logits are already small (SD about 0.25). Class-mean init and ETF heads hurt.
- **Init-norm and gain ideas:** conv weights reach norm equilibrium within about 40 steps, consistent with init gain being neutral.
- **DCT frequency re-allocation** (low-pass weighted): no mechanism with a clear sign, because high-frequency filter + max-pool already works as an energy detector.
- **Gradient centralisation:** it is cheap, but nothing in the evidence points to it.

## Rules check
All nine hypotheses are rule-compliant:
- Every lever is a scalar hyperparameter, a mathematical construction with per-trial seeded randomness, or a data-derived value computed in timed `prepare`/`train`.
- Nothing carries learned or pretrained state into trials, adds test-time augmentation, or adapts at evaluation.

Key files:
- `research/lab_recipe/config.py`
- `research/lab_recipe/train.py` (`make_optimisation`, `fit`)
- `research/lab_recipe/model.py` (`reset_model`, `structured_init_`, `init_whitening`, `AirbenchNet._head`)
- `research/sweeps/s50-new-base-round.toml` (S50, still running)

The `context7` and `huggingface-skills` MCP servers need authorising, in claude.ai connector settings or with `/mcp` in an interactive session. This task didn't need them.
