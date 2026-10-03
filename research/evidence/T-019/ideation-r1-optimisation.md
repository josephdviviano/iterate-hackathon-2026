<!-- Frozen report of agent acd44e43a44f853d1 (ideation round 1, read-only agent; model output, not user input). -->

# Optimisation and learning-dynamics hypotheses for the CIFAR-100 speedrun

I read `research/strategy-coverage.md`, `research/PROPOSAL.md`, the TRACKER findings F-001 to F-032 and every file in `submissions/team_segal/*.py`. I modified no files and ran no jobs.

**Shared protocol for every hypothesis.** Screen with 20 seeds, the same seeds in every arm, at 8.25 epochs on the dev stack (it matches the A100 within ±0.05 pp, F-031). Promote an arm if it gains at least 0.10 pp, then confirm it on 40 fresh seeds and trim epochs. Exchange rate: 0.25 pp is about a quarter epoch, or about 3% of time (F-021). Any arm that costs time must also be timed on one A100 PCIe host, paired against the control.

**Evidence patterns these hypotheses use:**
- **Effective learning rate.** Weight decay ×0.5 costs 1.15 pp, lr 9.2 costs 0.4 pp, and lr 14.4 and weight decay ×2 are roughly neutral. Accuracy drops steeply when the effective (angular) learning rate falls.
- **Lookahead is essential** (−0.77 pp without it).
- **The network is very sensitive to resolution and grid mismatches** (F-009).
- **The regime is capacity-bound and regularisation-negative:** cutout, mixup and freezing all lose.
- **Gains must come from early and mid-run steps,** because the accuracy curve flattens above 8.5 epochs.

---

## 1. fp16 update quantisation: fp32 master weights or stochastic rounding (rank 1)
**Mechanism.** `make_model` converts parameters to `.half()`, and fused SGD then updates fp16 weights with fp16 momentum. Lookahead's lerp and copy-back also run in fp16. The round-off threshold for an fp16 weight is about 2.4–4.9e-4 of its value. Late in the run the schedule is at 0.07–0.3, so per-element updates are around 1e-3 relative. Round-to-nearest then quantises them coarsely and drops the small ones entirely, a bias towards no update. The tail of the schedule may therefore be partly inert. That would fit the finding that a final lr of 0 or 0.15 makes no difference (F-016). This is a first-principles argument; mixed-precision practice (Micikevicius et al. 2018) exists to avoid exactly this.

**Implementation.** In `train.py`, keep an fp32 master copy of the conv, head and whitening-bias parameters (about 13.0M parameters; BatchNorm is already fp32). Each step: cast the fp16 gradients to fp32, run SGD on the masters, copy back to fp16 with `_foreach_copy_`, and run lookahead on the masters. A single `torch.compile`-d step function fuses this. Extra traffic is about 130 MB per step, roughly 0.07 ms, or under 0.5% of time.

**Experiment.**
- First a zero-cost diagnostic: one control run that logs the lost update mass, ‖Δ_intended − Δ_realised‖/‖Δ_intended‖, by step.
- Kill rule: reject without running arms if less than 5% is lost over the last 30% of steps.
- Otherwise compare control with fp32 masters on 20 seeds.

**Expected effect.** +0.05 to +0.2 pp. Reject if the gain is below +0.04 pp.

**Compliance.** All state is created and reset inside the trial.

## 2. Clean tail: turn augmentation off at the end (rank 2)
**Mechanism.** He et al. (2019, "Data augmentation revisited", arXiv 1909.09148) report gains on CIFAR from refining on un-augmented data at the end, which closes the gap between the augmented training distribution and the clean test distribution. F-009 shows this network is unusually sensitive to train/test mismatch. Translation is still needed early (translate 0 for the whole run costs 1.3 pp), but the reflect-padded shifts may hurt during the final low-lr steps. The change is free and slightly faster because the crop is skipped.

**Implementation.** In `TrainingStream.epoch` (`data.py`), add a `clean_tail_epochs` field to `config.py`. For epochs at or beyond that point, index `self.base[..., r:-r, r:-r]` instead of calling `batch_crop`, and keep the flip.

**Experiment.** Clean final epoch (epoch 8, the last 0.25 epoch) versus clean epochs 7–8 (the last 1.25 epochs) versus control, 20 seeds.

**Expected effect.** +0.05 to +0.2 pp. Reject if both arms are ≤ 0.

**Compliance.** Training data only.

## 3. Self-distillation from stored predictions (PS-KD style, no extra compute) (rank 3)
**Mechanism.** Label smoothing at 0.1, 0.2 and 0.3 is flat (F-022), so the amount of uniform smoothing does not matter. Instance-specific soft targets add information uniform smoothing cannot: which classes each image resembles, which matters with 100 classes. Kim et al. (ICCV 2021, PS-KD) and Shen et al. (CVPR 2022, DLB) report +1 to 2 pp on CIFAR-100. This differs materially from the dormant X-003 R8, which distilled from the lookahead slow weights and so needs an extra teacher forward pass. Here the targets come from the forward pass already made on that image in the previous epoch, so there are no extra FLOPs.

**Implementation.**
- Have `TrainingStream.epoch` also yield `idx`.
- Allocate a buffer `P = zeros(50000, 100, fp16)` inside the trial.
- After each forward: `P[idx] = outputs.detach().softmax(-1)`.
- New loss: (1−α)·CE_LS(y) + α·CE(P_prev[idx]).
- Ramp α from 0 in epoch 0 to α_max at the end.
- Snapshot the targets before overwriting them.

**Experiment.** α_max ∈ {0.3, 0.6} versus control, 20 seeds. Optional extra arm: label smoothing 0.2 annealed to 0 over the last 30%.

**Expected effect.** +0.1 to +0.3 pp. Reject if both α values are ≤ 0.

**Compliance.** The buffer is per-trial, built from training data only, and freed before `train` returns.

## 4. Break down the weight-decay sensitivity by parameter group (rank 4)
**Mechanism.** Weight decay ×0.5 at −1.15 pp is the largest sensitivity in the coverage matrix. It is unclear which parameters carry it:
- The convs are scale-invariant, because every conv feeds a frozen-scale BatchNorm, so for them weight decay only sets the effective lr.
- The head is not scale-invariant, so its weight decay sets the logit norm.
- The 1/6 logit-scale gain (F-019) changes three things at once: head effective lr (×2.25), initial temperature (×1.5) and the equilibrium logit norm.
- `bias_scaler=64` has never been swept.

**Implementation.** Split `make_optimizer` into head, conv, BatchNorm-bias and whitening groups, with per-group lr and weight-decay multipliers.

**Experiment.** Six arms, 20 seeds each: weight decay ×0.5 on the convs only; weight decay ×0.5 on the head only; head lr ×2; head lr ×0.5; bias_scaler 32; bias_scaler 128. Interpretation:
- If "head only" reproduces most of the −1.15 pp, exploit head weight decay and temperature.
- If "convs only" does, go to hypothesis 5.

**Expected effect.** At least one arm gains ≥ +0.1 pp, with about 30% probability. Reject if every arm is within ±0.08 pp of control.

**Compliance.** Scalar hyperparameters only.

## 5. Rotational-equilibrium warm start (rank 5)
**Mechanism.**
- For a scale-invariant weight, the effective lr is η/‖w‖².
- Under SGD with weight decay, the angular update rate settles to about √(2ηλ/(1+μ)) per step, with a time constant of about 1/(2ηλ). That is roughly 100 steps during warm-up, about 25% of this 396-step run.
- Before equilibrium, layers rotate too slowly and unevenly (Kosson, Messmer & Jaggi, ICML 2024; van Laarhoven 2017; Li & Arora 2020).
- The lr and weight-decay asymmetry in the shared evidence is the signature of a run that sits on the low side of its angular-lr optimum early on.

**Implementation.** In `reset_model` (`model.py`), multiply every `Conv` weight, including its identity (dirac) block, by `init_gain` c < 1. Forward outputs are unchanged and the starting effective lr rises by 1/c². Follow-up, if positive: per-filter projection onto a fixed norm with an explicit angular-lr schedule (RV-SGD), and a shorter warm-up.

**Experiment.** c ∈ {0.5, 0.7} at the current warm-up, plus c = 0.6 with `lr_peak_frac` 0.12, 20 seeds. Log per-layer ‖W‖ by step.

**Expected effect.** 0 to +0.15 pp. Reject if accuracy does not rise as c falls.

**Compliance.** Initialisation scalar only.

## 6. Loss shock at the 20→32 px switch (rank 6)
**Mechanism.** At the switch the grids change from 19→9→4→2 to 31→15→7→3. The global max goes from 4 to 9 positions, which raises the expected Gaussian maximum by about 0.45σ. Every stage's spatial statistics shift too. F-009 (28 px training, 32 px evaluation gives 69%) shows how large such shifts are. Steps spent recovering are wasted.

**Implementation and experiment.**
- First a diagnostic: log per-step training loss and accuracy around step 198 in 3 seeds.
- If loss jumps and takes more than about 15 steps to recover, test two remedies against control, 20 seeds:
  - (a) a stochastic blend over steps 178–218 that picks 32 px with a ramped probability, using the two graphs that are already compiled;
  - (b) momentum buffer ×0.5 at the switch. This is a partial reset; the competitor tested and rejected only a full restart.

**Expected effect.** +0.05 to +0.1 pp. Kill rule: reject if recovery takes ≤ 5 steps.

**Compliance.** Fine.

## 7. Fixed simplex-ETF classifier (rank 7)
**Mechanism.** Neural-collapse results (Yang et al., NeurIPS 2022; Hoffer et al., ICLR 2018, "Fix your classifier") show a fixed, maximally separated head matches a learned head on CIFAR-100. In a short run it removes the lag between head and features: the features chase fixed targets from step 1.

**Implementation.** In `Net` (`model.py`): `head.weight = √(K/(K−1))·(U(I − 11ᵀ/K))ᵀ`, with U a 640×100 orthonormal matrix from QR of a Gaussian drawn from a fixed constant seed. Freeze it and add a learnable per-class bias. The features are mostly non-negative after the max-pool, so class-specific offsets need correcting.

**Experiment.** Logit scale ∈ {1/6, 1/3, 2/3}, 10 seeds.

**Expected effect.** −0.2 to +0.2 pp. Reject if the best arm is below control.

**Compliance.** A mathematical constant, not a learned tensor.

## 8. Annealed soft max-pool in the 20 px phase only (rank 8)
**Mechanism.** The global max sends gradient to only one of the 4 terminal positions per channel, which slows early learning in stage 3. Log-sum-exp pooling (Pinheiro & Collobert 2015) gives dense gradients and becomes the exact max as τ→0. This differs from the rejected max+mean pool, which was a permanent mix that also applied at test time. Here only the 20 px graph changes, so the 32 px graph and evaluation stay exact max.

**Implementation.** In `Net.forward`: `τ·logsumexp(x/τ) − τ·log N`, computed in fp32, with τ held in a buffer tensor to avoid recompiles. τ anneals to near 0 by the switch.

**Experiment.** τ₀ ∈ {0.5, 1} (in BatchNorm-unit scale), 10 seeds.

**Expected effect.** 0 to +0.1 pp. Reject if ≤ 0.

**Compliance.** Evaluation is a pure exact max.

## 9. Sharpening targets at the end (rank 9)
**Mechanism.** In an underfitting regime, smoothing helps early optimisation but caps margins late. Anneal label smoothing from 0.2 to 0 and the logit scale up by ×1.25 over the last 25%. Support is weak (the flat label-smoothing levels); run this only as a free arm alongside hypothesis 3.

**Expected effect.** ≤ +0.1 pp.

---

**Where to start.** Hypotheses 1, 2 and 6 begin with diagnostics or one-line arms that cost almost no GPU time. Hypotheses 3 and 4 are the largest plausible gains, at no time cost. If two of them each confirm +0.1 to +0.15 pp, together they would fund about a quarter epoch, roughly −0.15 to −0.18 s.

**Files involved:**
- `submissions/team_segal/train.py` (`fit`, `make_optimizer`, `Lookahead`)
- `submissions/team_segal/data.py` (`TrainingStream.epoch`)
- `submissions/team_segal/model.py` (`make_model`, `reset_model`, `Net.forward`)
- `submissions/team_segal/config.py`

The lab copy at `research/lab_recipe/train.py` already has `refit_head`, which belongs to the lead's head-refit idea.

**Note for the user.** The `context7` and `huggingface-skills` MCP servers need authorising (via `claude mcp` or `/mcp` in an interactive session). I did not need them for this task.
