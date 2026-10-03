# CIFAR-100 speedrun entry

> Living human-readable programme record. Generated sections are refreshed from canonical state;
> edit only the human feedback and steering region below.

## Human feedback and steering

<!-- writing-tools:human:start -->
### Open feedback

<!-- Add feedback, questions, or steering here. Use `**Blocking:** yes` on its own line only when work must pause. -->

### Responses and resolved feedback

<!-- Preserve the original feedback here and append the response, decision, or resulting change. -->
<!-- writing-tools:human:end -->

<!-- writing-tools:generated:start -->
## Current position

- **State:** needs_design
- **Mission:** Submit a rule-compliant CIFAR-100 speedrun entry whose official 40-seed evaluation on one NVIDIA A100 80GB PCIe qualifies (mean top-1 at least 75%) at the lowest mean prepare+train time the team can demonstrate, with recorded evidence for every adopted and rejected technique.
- **Root question:** Which compliant recipe minimises mean A100 PCIe prepare+train time while keeping the official 40-seed mean top-1 at or above 75% with a qualification risk of about 1% or less?
- **Why:** Programme evidence or cross-artifact integrity is invalid; repair it before selecting more work.
- **Next:** Continue T-019 (Structural exploration with agent ideation rounds (X-006)).
- **Open human feedback:** none

## Work completed and underway

| Task | Type | State | Outcome or purpose |
| --- | --- | --- | --- |
| [T-001](tasks/T-001-local-accuracy-stack-on-the-blackwell-gpus.md) — Local accuracy stack on the Blackwell GPUs | delivery | completed | Dev stack torch 2.7.1+cu128 runs the official harness on real data on both local GPUs; setup script and freeze reproduce it from a fresh worktree. |
| [T-002](tasks/T-002-parametrised-airbench-lineage-recipe-substrate-in-the-team-folde.md) — Parametrised airbench-lineage recipe substrate in the team folder | delivery | completed | Parametrised recipe passes CPU contract tests (reset, invalid params, defaults, harness smoke for three variants) on the pinned stack and trains on real data (default 69.5% single-view, eager evaluation). |
| [T-003](tasks/T-003-declarative-multi-gpu-sweep-runner-collation-and-runbook.md) — Declarative multi-GPU sweep runner, collation and runbook | delivery | completed | Sweep runner dispatches across both GPUs through the harness, resumes only interrupted configs, records failures, and collates reproducibly; runbook replayed from a fresh worktree. |
| [T-004](tasks/T-004-a100-pcie-timing-calibration-and-cross-stack-accuracy-agreement.md) — A100 PCIe timing calibration and cross-stack accuracy agreement | exploration | completed | A100 80GB PCIe: 5.76 s (512), 6.17-6.26 s (640, default compile), 6.12 s with max-autotune; cross-stack accuracy agrees within 0.1 pp over 120 paired seeds. |
| [T-005](tasks/T-005-single-view-width-by-epochs-frontier-p1.md) — Single-view width by epochs frontier (P1) | exploration | completed | Frontier measured over 4 shapes x 4 epoch counts x 5 seeds plus a ResNet-9 arm; airbench96 shape reaches 75.37% at 10 epochs, 2x crosses at 13.7 epochs; capacity-bound (F-003). |
| [T-006](tasks/T-006-select-the-base-regime.md) — Select the base regime | decision | completed | Regime settled on 128/384/640 at 8.25 epochs from the frontier and A100 timings (D-008). |
| [T-007](tasks/T-007-add-on-levers-at-the-selected-base-p2.md) — Add-on levers at the selected base (P2) | exploration | cancelled | Superseded by T-013: probes S4-S15 tested Muon, resizing, selection, batch size, regularisation, compile and further levers at the climbed base (F-007 to F-019); its premise of a separate add-on pass after an A100 regime decision no longer holds because the climb ran on the local proxy under D-001. |
| [T-008](tasks/T-008-adopt-or-reject-add-on-levers.md) — Adopt or reject add-on levers | decision | cancelled | Superseded by T-013 decisions D-003 to D-007, which record adopted and rejected levers with evidence and reopening conditions. |
| [T-009](tasks/T-009-converge-and-simplify-the-final-submission.md) — Converge and simplify the final submission | delivery | cancelled | Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance. |
| [T-010](tasks/T-010-official-equivalent-40-seed-qualification-on-the-a100-pcie.md) — Official-equivalent 40-seed qualification on the A100 PCIe | assurance | cancelled | Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance. |
| [T-011](tasks/T-011-fresh-context-compliance-review-and-independence-checks.md) — Fresh-context compliance review and independence checks | assurance | cancelled | Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance. |
| [T-012](tasks/T-012-open-the-upstream-pull-request-after-approval.md) — Open the upstream pull request after approval | delivery | cancelled | Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance. |
| [T-013](tasks/T-013-local-proxy-hill-climb-to-convergence-x-002.md) — Local-proxy hill climb to convergence (X-002) | exploration | completed | Converged after S3-S19: 128/384/640 depth-3, translate 2, logit scale 1/6, 20/32 px resizing, compiled; 40-seed fresh means 75.33% at 8.5 and 75.43% at 8.75 epochs (AC-03), budget 8.25 epochs (75.26%) selected under F-005 (D-007); local compiled proxy 5.38 s vs 9.3 s for the P1 best cell. |
| [T-014](tasks/T-014-converge-the-climbed-recipe-into-submission-defaults.md) — Converge the climbed recipe into submission defaults | delivery | completed | Team folder converged and simplified: defaults are the T-013 recipe (75.21% over 10 fresh seeds at 5.18 s local), exploration paths moved to research/lab_recipe, training bit-identical to the lab substrate, compile and fused SGD CUDA-only, README written; 23 contract tests pass on the pinned stack. |
| [T-015](tasks/T-015-official-equivalent-40-seed-qualification-on-the-a100-pcie.md) — Official-equivalent 40-seed qualification on the A100 PCIe | assurance | completed | Qualifies on A100 80GB PCIe: 75.272% over 40 fresh seeds at 6.027 s per trial, risk about 2e-7; telemetry recorded. |
| [T-016](tasks/T-016-fresh-context-compliance-review-and-independence-checks.md) — Fresh-context compliance review and independence checks | assurance | ready | Review the frozen converged candidate against every RULES.md section 3 bullet without implementation narrative, and run repeat-seed, reordered-seed and fresh-process independence checks. |
| [T-017](tasks/T-017-open-the-upstream-pull-request-after-approval.md) — Open the upstream pull request after approval | delivery | proposed | After a recorded team-lead approval, create a clean branch from upstream main containing only the team folder and open the pull request. |
| [T-018](tasks/T-018-coverage-pass-every-lineage-competitor-and-novel-strategy-x-003.md) — Coverage pass: every lineage, competitor and novel strategy (X-003) | exploration | completed | Every lineage, competitor and representation strategy has a disposition; only flatten-max pooling and max-autotune CUDA graphs survived (about 3% faster at equal accuracy) and are now defaults; the momentum candidate failed fresh-seed confirmation. |
| [T-019](tasks/T-019-structural-exploration-with-agent-ideation-rounds-x-006.md) — Structural exploration with agent ideation rounds (X-006) | exploration | in_progress | Explore structural vectors beyond the converged recipe: ensembles within the untimed evaluation budget, closed-form head refit, schedule shapes, lookahead dynamics, data ordering, head geometry, in-run soft targets, pooling geometry and A100 systems levers; run agent ideation at the start and at each convergence point, and adopt only levers confirmed on fresh seeds and same-host A100 timing. |

## Issues identified

### Programme integrity and ownership

- T-003: E-004 has changed since verification: research/tests/test_sweep.py
- R-006: E-003 has changed since verification: research/tests/test_sweep.py

### F-001 — resolved, material

- **Observation:** The default airbench94-equivalent recipe (64/256/256, 10 epochs, batch 1024, no TTA) reaches 69.4-69.7% single-view CIFAR-100 accuracy on the dev stack (7 trials over eager and compiled runs); the smoke sweep gives 59.8% at 4 epochs and 43.1% at 2 epochs for the same width.
- **Interpretation:** The airbench94 shape is about 5.5 pp short of 75% at 10 epochs, versus 93.3% single-view on CIFAR-10 at the same budget; reaching 75.3% needs more epochs, more width, or both, so the P1 grid range (10-24 epochs, widths 1x-2x plus the airbench96 shape) is appropriately placed. The observation does not yet separate H2 from H3.
- **Decision consequence:** Launch P1 as specified; if no cell reaches 75.3% by 24 epochs, extend epochs or width before the regime decision.
- **Resolution:** Recorded as the P1 starting anchor.

### F-002 — resolved, contextual

- **Observation:** Both local RTX PRO 6000 GPUs run other projects' jobs at 100% utilisation, 300 W power cap and 87-88 C; a compiled default run took 14.8-23.6 s on GPU 1 versus 8.0-9.0 s eager on GPU 0, with no recompilation logged.
- **Interpretation:** Local wall times reflect contention and power capping, not the recipe; exploration throughput depends on the other workloads.
- **Decision consequence:** Keep one slot per GPU, never compare local times, and take all timing from the A100 (T-004).
- **Resolution:** Runbook section 4 sets one slot per device; local times are labelled non-evidence.

### F-003 — resolved, material

- **Observation:** P1 (5 seeds per cell, single-view, dev stack): the airbench94 shape plateaus at 71.5% (1x) and 74.0% (1.5x) by 24 epochs; 2x crosses 75.3% at an interpolated 13.7 epochs (12.0 s local proxy); the airbench96 shape (128/384/512, three convs per block with a residual, translate 4) reaches 75.37% at 10 epochs (9.3 s) and 76.9% at 24.
- **Interpretation:** Single-view 75% on CIFAR-100 is capacity-bound: extra epochs barely help narrow nets, while width and depth move accuracy several points. H2 (airbench tricks reach the target within 20 epochs) and H3 (minimum-time point at airbench96-class capacity) are supported; the best cell lies at the grid edge, so the optimum may sit below 10 epochs.
- **Decision consequence:** Climb from the airbench96 shape: probe fewer than 10 epochs and nearby depth-3 shapes before optimiser and resolution levers.
- **Resolution:** Base for the hill climb (T-013) is the airbench96 shape.

### F-004 — resolved, contextual

- **Observation:** The ResNet-9 reference arm reached only 57.96% (sd 0.76 pp) at 40 epochs, versus the organisers' 75.36% for a ResNet-9 baseline at the same budget.
- **Interpretation:** The arm reused airbench's optimiser parametrisation (lr 11.5 per 1024, BN-bias lr x64, frozen-BN momentum) on a ReLU network with trainable BatchNorm, so the shortfall reflects mistuned hyperparameters, not the architecture. Retuning is not decision-bearing: ResNet-9 costs 113.8 TFLOP per epoch and needs about 40 epochs (organiser evidence), roughly 3.4x the FLOPs of the airbench96 shape at 10 epochs.
- **Decision consequence:** Reject ResNet-9 as a base (H1) on FLOP grounds; do not spend GPU time retuning the arm.
- **Resolution:** H1 rejected as base; arm retained only as a recorded negative.

### F-005 — resolved, material

- **Observation:** Pooled within-cell sd of single-view accuracy over the 16 airbench P1 cells is 0.245 pp (cells range 0.08-0.37 pp).
- **Interpretation:** H6 is supported: with sd near 0.25 pp, a development mean of 75.2% over 40 seeds keeps the estimated official qualification risk under 1% (about 0.4%).
- **Decision consequence:** Use 75.3% as the conservative climbing target and 75.2% over at least 40 seeds as the final development criterion.
- **Resolution:** Margin set; reconfirm sd at the final recipe.

### F-006 — resolved, material

- **Observation:** S3 (5 seeds per cell, GPU 1): the airbench96 shape with translate 4 reaches 75.3% at 9.68 epochs (9.02 s proxy); with translate 2 it reaches it at 8.89 epochs (8.28 s), +0.41 pp at 8 and +0.48 pp at 10 epochs; 128/512/768 reaches it at 7.13 epochs but 9.43 s; 96/256/384 tops out at 73.9% by 10 epochs.
- **Interpretation:** H7 is supported through the budget and augmentation dimensions rather than a new shape: lighter translation helps short CIFAR-100 runs (an H9-style regularisation effect), and the wider 128/512/768 shape saves epochs but not time. The 128/384/512 depth-3 shape remains the efficient capacity point.
- **Decision consequence:** Climb levers at 128/384/512, three convs, translate 2, bracketing the target at 8 and 10 epochs (S4); test translate 0-1 and the wider shape again at translate 2.
- **Resolution:** Base updated for S4.

### F-007 — resolved, material

- **Observation:** S4 at the D-002 base (5 seeds per arm, 8 and 10 epochs): resizing 24 then 32 px at half-way reaches 75.3% at 9.66 epochs and 7.05 s proxy versus 8.25 s for control (-15%); translate 1 (8.28 s) and label smoothing 0.1 (8.30 s) tie with control; translate 0 (-1.3 pp), lookahead off (-0.8 to -0.9 pp), batch 2048 (-0.6 pp) and Muon at batch 1024 or 2000 (-0.8 to -1.4 pp, about 10% slower per step) lose; batch 1536 is slightly slower (8.47 s); 128/512/768 at translate 2 reaches 76.42% at 8 epochs. The control arms reproduce S3 exactly (74.858%, 75.854%).
- **Interpretation:** H5 is supported (resolution); H4 is disconfirmed for Muon as ported with airbench94_muon hyperparameters; H8 is disconfirmed; H9 is partly supported (light translation helps, label smoothing is flat, lookahead is necessary). Identical control means across sweeps confirm that results reproduce bit-for-bit for fixed code and seeds on this stack.
- **Decision consequence:** Adopt progressive resizing and tune its schedule; keep SGD with lookahead, batch 1024, label smoothing 0.2, translate 2; retest the wide shape with resizing (S5).
- **Resolution:** Resizing adopted provisionally (D-003); Muon parked as dormant pending a retune if the climb plateaus.

### F-008 — resolved, material

- **Observation:** S5 (5 seeds per cell): resize schedules at the base shape reach 75.3% in 6.89 s (20 px then 32 at half-way), 7.20 s (16/24/32), 7.32 s (24/32 at half-way; S4 estimate 7.05 s), 7.39 s (24/28/32); 24 px for 70% never reaches 75.3% by 12 epochs; 128/512/768 reaches it in 8.56 s at full resolution and 7.34 s with 24/32 resizing.
- **Interpretation:** Within the resizing family proxy time has plateaued: differences are within the about 5% resolution of 5-seed cells (0.1 pp accuracy SE is about 0.4 epochs). Long low-resolution phases cost too much accuracy, and the wider shape does not beat the base once epochs are cheap.
- **Decision consequence:** Fix the base at 20 px then 32 px at half-way, about 10.6 epochs; treat future gains under about 5% as noise unless confirmed with 10 or more seeds.
- **Resolution:** Base updated in S6 (20/32 schedule).

### F-009 — resolved, material

- **Observation:** S6 at the 20/32 px base (5 seeds per arm): control 75.47% at 10 epochs (6.68 s) and 75.62% at 12 (8.07 s); compile plus fused SGD runs 11-12% faster per epoch (5.92 s, 7.17 s) at 75.06% and 75.61%; freezing stage 1 at 60% costs 0.7-0.9 pp (74.61%, 74.95%) for 15% less time; freezing stages 1 then 2 costs 1.6-1.7 pp; ending at 28 px and evaluating at 32 px drops accuracy to 69.0-69.4%.
- **Interpretation:** H14 is supported for time (about 11% per epoch); its 0.41 pp dip at 10 epochs (about 2.9 combined SE) is unresolved between noise, compile numerics and fused SGD. H11 is disconfirmed at these freeze points: early stages still learn late in short runs. H12 is strongly disconfirmed: the network does not tolerate a 28 to 32 px train-test gap, plausibly through BatchNorm statistics and the global max-pool scale. The control curve is nearly flat beyond 10 epochs, so epochs to target are sensitive to small accuracy shifts.
- **Decision consequence:** Reject freezing and FixRes endings; separate compile from fused SGD with more seeds before adopting either; fine-tune the epoch count near 9-10 with at least 10 seeds.
- **Resolution:** Freezing and FixRes rejected; compile decomposed in S8.

### F-010 — resolved, material

- **Observation:** S7 at the base (5 seeds): in-run selection keeping the highest-loss 50% of each batch reaches only 64.5%, 66.2% and 68.1% at 10, 12 and 14 epochs (sd up to 1.9 pp); keeping 70% reaches 73.3% and 74.1% at 10 and 12 epochs; the in-sweep control reaches 75.33% and 75.85%.
- **Interpretation:** H13 is disconfirmed for short CIFAR-100 runs: training only on the hardest examples starves a 10-epoch, 100-class run of the easy examples it still needs, and the high variance suggests instability. airbench96_faster's gain came from a 45-epoch CIFAR-10 regime.
- **Decision consequence:** Reject in-run selection; do not spend more GPU time on selection variants unless the climb plateaus with long budgets.
- **Resolution:** Selection rejected (H13).

### F-011 — resolved, material

- **Observation:** S7 (5 seeds): max-autotune compilation with fused SGD is 13% faster per epoch (5.81 s vs 6.71 s at 10 epochs) but 0.28 pp and 0.36 pp lower at 10 and 12 epochs (75.05%, 75.49% vs 75.33%, 75.85%); lr 9.2 lowers accuracy (74.92%, 75.44%) and lr 14.4 ties the default (75.36%, 75.56%). The compile arm still used dynamic-shape recompilation (fixed afterwards with dynamic=False).
- **Interpretation:** The compiled-plus-fused arms are lower in both S6 and S7 at matched epochs, so the deficit looks systematic rather than noise; its source (fused SGD versus compiled numerics) is unresolved. H15 is rejected: the default learning rate is already near-optimal for the base.
- **Decision consequence:** Keep lr 11.5; resolve the compile deficit in S8 (10 seeds, compile and fused separated, dynamic=False) before adopting compilation for timing.
- **Resolution:** lr fixed at default; compile decision deferred to S8.

### F-012 — resolved, material

- **Observation:** S9 (5 seeds): batch 512, 768 and 1024 reach 75.3% in 6.65, 6.74 and 6.67 s; widths 128/384/640 reach 75.42% at 9 epochs and 75.72% at 10 (time to target <= 6.50 s); 128/448/512 needs 7.50 s, 160/384/512 and 96/384/512 do not reach 75.3% by 10 epochs; Muon with lookahead and the standard 1/9 head produced non-finite logits in all four arms (lr 0.12 and 0.24).
- **Interpretation:** Batch size is flat. Capacity added to the last stage, which runs at 4x4 to 8x8 resolution, buys accuracy most cheaply; earlier-stage width does not. Muon without head normalisation diverges in fp16, so Muon is rejected for this programme (H4 rejected).
- **Decision consequence:** Climb last-stage width (S12); keep batch 1024 and SGD.
- **Resolution:** Last-stage width explored in S12; Muon rejected.

### F-013 — resolved, material

- **Observation:** S8 (10 fresh seeds 100-109, dynamic=False): at 9 and 10 epochs, eager 74.64% and 75.27%, fused SGD 74.79% and 75.27%, compile 74.77% and 75.21%, compile plus fused 74.79% and 75.43% (SE 0.05-0.12 pp); compilation lowers per-trial time by about 12% (6.06 to 5.32 s at 9 epochs, 6.73 to 5.90 s at 10); fused SGD does not change speed; compile adds 3.5-10 s of untimed build.
- **Interpretation:** H14 is supported: compilation is a pure ~12% speed-up with accuracy unchanged within noise. The S6/S7 accuracy deficits were 5-seed noise, possibly compounded by dynamic-shape recompilation. On fresh seeds the base sits at 75.27% after 10 epochs, just under the climbing target.
- **Decision consequence:** Use compile (dynamic=False, default mode) with fused SGD for all timing; rank accuracy-changing levers with the eager proxy, which compile scales uniformly; the base needs a little more capacity or epochs for margin.
- **Resolution:** Compile adopted (D-004).

### F-014 — resolved, material

- **Observation:** S12 (5 seeds, eager proxy): time to 75.3% is 6.29 s for 128/384/640 (8.70 epochs), 6.45 s for 128/320/768 (8.96), 6.52 s for 128/384/768 (8.36) and above 7.26 s for 128/384/1024; the 128/384/512 control does not reach 75.3% by 9 epochs (74.68%).
- **Interpretation:** Last-stage width is the efficient lever up to about 640 channels: it cuts epochs enough to more than pay for its cost, beyond which per-epoch cost dominates. This also restores margin that the S8 fresh-seed control lacked.
- **Decision consequence:** Adopt 128/384/640 as the base (D-005) and fine-tune epochs near 8.5-9 with compiled timing and 10 seeds.
- **Resolution:** Base widths updated to 128/384/640.

### F-015 — resolved, material

- **Observation:** S11 (5 seeds): pooling before the widening conv in stages 2-3 cuts time 22% but drops accuracy to 73.13% and 73.52% at 10 and 12 epochs (control 75.33%, 75.62%); pooling first in all stages drops to 70.3-70.8%; reinvesting in 128/512/768 with pool-first reaches only 74.05% and 74.36%; a depth-2 first stage reaches 75.02% at 10 epochs for 7% less time (12-epoch cell pending).
- **Interpretation:** The widening convs need full spatial resolution before pooling; their FLOPs are productive, so FLOP reallocation through pool-first is rejected. The depth-2 first stage is within noise of a time-neutral trade and does not change the base.
- **Decision consequence:** Keep conv-then-pool in every stage and three convs per stage; spend remaining effort on low-level execution (pooling implementation, compile) and epoch tuning.
- **Resolution:** Pool-first rejected; stage depths unchanged.

### F-016 — resolved, material

- **Observation:** S10 at the 128/384/512 base (5 seeds, 9 and 10 epochs): control 74.68% and 75.33%; lr peak 0.1 or 0.35, final lr 0 or 0.15, and whitening-bias epochs 1 or 6 all fall within 0.3 pp of control (best: whitening-bias 6 epochs, 6.61 s vs 6.69 s to target); a short final 32 px phase (20/28/32) and a 16 px start lose 1.2-1.8 pp.
- **Interpretation:** Schedule and optimiser hyperparameters are at a plateau for this base: the airbench defaults transfer. Resolution schedules that shorten the final full-resolution phase lose accuracy, consistent with F-009.
- **Decision consequence:** Freeze schedule hyperparameters at their defaults; stop hyperparameter climbing and converge on epochs and execution.
- **Resolution:** Schedule dimension saturated.

### F-017 — resolved, material

- **Observation:** S13 (10 fresh seeds 200-209, compiled, fused SGD, 128/384/640): 75.20% at 8.5 epochs (5.52 s), 75.42% at 9.0 (5.84 s), 75.47% at 9.5 (6.17 s); per-cell sd 0.19-0.32 pp; interpolated 75.3% crossing at 8.74 epochs and 5.67 s compiled proxy.
- **Interpretation:** The D-005 base holds on fresh seeds and, compiled, reaches the climbing target about 39% faster than the P1 best cell (9.3 s eager). The accuracy curve flattens above 9 epochs, so the R-001 margin is cheapest to buy between 8.75 and 9 epochs.
- **Decision consequence:** Confirm the final recipe with 40 fresh seeds at 8.75 and 9.0 epochs; select the shortest budget whose 40-seed mean is at least 75.2%.
- **Resolution:** Final confirmation follows (S15).

### F-018 — resolved, contextual

- **Observation:** S14 (5 seeds, 128/384/640, 9 epochs): pool_impl amax matches indexed max-pool accuracy in eager mode (75.47% vs 75.52%) but is 62% slower (10.71 s vs 6.62 s); compiled amax produced non-finite logits and failed; compiled indexed max-pool reaches 75.33% in 5.81 s. Competitor run codex c7 also failed with non-finite logits when its max+mean head (amax) was compiled.
- **Interpretation:** The reshape-plus-amax pooling is exact on CPU but eager CUDA reductions over channels_last tensors are slow, and compiled amax diverges on this stack, plausibly an Inductor lowering problem at training shapes. The pooling profile cost cannot be removed this way.
- **Decision consequence:** Keep indexed max-pool (pool_impl torch); do not compile amax-based pooling or heads.
- **Resolution:** amax pooling rejected.

### F-019 — resolved, material

- **Observation:** S15 (10 fresh seeds 300-309, compiled 128/384/640): logit scale 1/6 reaches 75.28% at 8.5 and 75.61% at 9.0 epochs versus 75.00% and 75.27% at the default 1/9; scale 2/9 reaches 75.07% and 75.15%; time is unchanged; the 75.3% crossing moves to 8.53 epochs and 5.53 s. The competitor's independent 10-seed run (codex c7) found the same 1.5x gain worth +0.38 and +0.19 pp at 9 and 10 epochs.
- **Interpretation:** A sharper softmax (1.5x logits) speeds short-run convergence with label smoothing 0.2; the effect replicates across two codebases and seed sets. Seed-set means differ by about 0.15 pp (S13 control 75.42% vs S15 control 75.27% at 9 epochs), so the final budget must be set from many fresh seeds.
- **Decision consequence:** Adopt scaling_factor 1/6 (D-006) and confirm with 40 fresh seeds at 8.75 and 9.0 epochs (S16).
- **Resolution:** Logit scale adopted.

### F-020 — resolved, material

- **Observation:** S16 (40 fresh seeds 400-439, compiled, 128/384/640, logit scale 1/6): 8.75 epochs gives 75.43% mean (sd 0.235 pp, worst seed 74.96%) in 5.69 s local proxy; 9.0 epochs gives 75.48% (sd 0.243 pp) in 5.85 s; all 80 trials completed.
- **Interpretation:** The climbed recipe meets the 75.2% development criterion with 0.23 pp to spare at 8.75 epochs; the estimated official qualification risk is far below 1% (margin about 8 combined SE). The spare margin means a shorter budget may also qualify.
- **Decision consequence:** Search 8.0-8.5 epochs with 40 fresh seeds (S18) and select the shortest budget meeting 75.2%.
- **Resolution:** Confirmation passed; shortening probed in S18.

### F-021 — resolved, material

- **Observation:** S18 (40 fresh seeds 600-639, climbed recipe): 8.0 epochs 75.02% (sd 0.29 pp, 5.21 s), 8.25 epochs 75.26% (sd 0.24 pp, 5.38 s), 8.5 epochs 75.33% (sd 0.22 pp, 5.52 s); with S16, 8.75 epochs 75.43% (5.69 s).
- **Interpretation:** 8.25 epochs is the shortest budget meeting the 75.2% development criterion; its 0.26 pp margin over 75% is about 5 combined standard errors, so dev-stack qualification risk is negligible. Accuracy rises about 0.25 pp per quarter-epoch below 8.5 and flattens above it.
- **Decision consequence:** Select 8.25 epochs (D-007) with 8.5 and 8.75 epochs as fallbacks if A100 confirmation (T-004, T-010) falls below 75.2%.
- **Resolution:** Final local budget selected.

### F-022 — resolved, material

- **Observation:** S17 (10 fresh seeds 500-509, climbed recipe, 8.75 epochs): label smoothing 0.1 / 0.2 / 0.3 give 75.35% / 75.41% / 75.52% at equal time; switching to 32 px at 40% gives 75.84% in 6.19 s, at 60% gives 75.01% in 5.18 s, versus 75.41% in 5.69 s at 50%.
- **Interpretation:** Label smoothing is flat around 0.2. The switch point trades accuracy for time: 60% lies on the same curve as trimming epochs (S18: 8.0 epochs 75.02% in 5.21 s), while 40% sits about 0.3 pp above that curve at equal time and may reach the D-007 level sooner.
- **Decision consequence:** Keep label smoothing 0.2; test the 40-45% switch at 7.5-8 epochs before declaring convergence (S19).
- **Resolution:** Switch-point follow-up in S19.

### F-023 — resolved, material

- **Observation:** S19 (20 fresh seeds 700-719): the 50% switch at 8.25 epochs gives 75.22% in 5.38 s; a 45% switch at 8.0 epochs gives 75.22% in 5.43 s; a 40% switch gives 75.09% at 7.5 epochs (5.27 s) and 75.36% at 8.0 epochs (5.67 s), interpolating to about 5.46 s at 75.22%.
- **Interpretation:** At matched accuracy the 50% switch is fastest; earlier switches buy accuracy at a time cost that is no better than adding epochs. With S10, S15 and S17 this exhausts the accessible levers: the climb has converged at the D-005 to D-007 recipe. Pooled over S18 and S19, 60 fresh seeds at 8.25 epochs average about 75.25%.
- **Decision consequence:** Declare X-002 saturated, close T-013, and converge the recipe into submission defaults (T-014); remaining uncertainty is cross-stack transfer and A100 timing (T-004, T-015).
- **Resolution:** Climb converged.

### F-024 — resolved, contextual

- **Observation:** The organisers' internal plan (CIFAR-100-speedrun-plan.md, deleted in upstream 25237e3 but present in its history) states that build may compile, autotune, allocate buffers and capture CUDA graphs on synthetic data; that prepare should hold whitening, model and optimiser resets and augmented-batch construction; and that organisers will red-team: whitening or augmentation in build, cached trained state, RNG manipulation, cooldown sleeps, modified evaluation, test-label access and smuggled weights. The calibration recipe is not in the history.
- **Interpretation:** The plan adds no hidden recipe but defines the review the organisers will apply. The team_segal recipe matches its intended use: synthetic-only warm-up in build, whitening and resets in prepare, augmentation drawn from a generator seeded by the trial seed, eager stateless evaluation. A fixed, tuned internal seed would fall under RNG manipulation and stays off-limits.
- **Decision consequence:** Use the plan's red-team list as the T-016 compliance checklist; test CUDA-graph compile modes on the A100 (m2).
- **Resolution:** Checklist adopted for T-016.

### F-025 — resolved, material

- **Observation:** S22 (10 seeds): the hiverge CIFAR-10 record's optimiser stack (Muon with hiverge NS coefficients, renormalisation every 2 to 17 steps, decoupled weight decay, lr 0.205, momentum 0.655 and 0.825, head_norm, label smoothing 0.09, whitening bias 0.2 epochs) on the converged architecture reaches 73.07% / 73.77% at 8.25 / 10 epochs with batch 1024, 72.49% / 73.56% with batch 1536, and 72.01% / 72.54% with Muon lr 0.15 / 0.30, versus 75.33% for the SGD control at 8.25 epochs.
- **Interpretation:** Muon is rejected for single-view CIFAR-100 in both ports (airbench94_muon, F-007 and F-012; hiverge here): it trails SGD with lookahead by 1.5-3.3 pp at every batch size, budget and lr tried. The hiverge stack was tuned for 10-way, TTA-scored, 64/256/256 training.
- **Decision consequence:** Close the optimiser dimension on SGD with lookahead.
- **Resolution:** Muon rejected.

### F-026 — resolved, material

- **Observation:** S24 (10 seeds 1300-1309; control 75.14% at 5.39 s): RepVGG-style 1x1 branches 75.19% at 6.51 s; PolyLoss-1 75.10%; squentropy 74.77%; superclass auxiliary loss 75.14% (weight 0.3, +5% time) and 73.25% with sd 1.2 pp (weight 1.0); ConvMixer-512/8 (k5, patch 2) 25.6% in 19.9 s. Progressive deepening crashed in fused SGD and is rerun in S25.
- **Interpretation:** R3, R4 and R5 are disconfirmed: train-time over-parameterisation buys nothing for its 21% cost, alternative losses do not beat smoothed cross-entropy, and ConvMixer's depthwise mixing neither learns in 8 epochs nor runs fast. R2 shows no gain at a mild weight and destabilises training at weight 1.0, so the hierarchy route is rejected and its rule question is moot.
- **Decision consequence:** Keep the airbench convnet, plain 3x3 convs and smoothed cross-entropy; finish R1 in S25.
- **Resolution:** R2-R5 rejected; R1 rerun.

### F-027 — resolved, material

- **Observation:** S25 (10 seeds 1300-1309, foreach SGD): control 75.06% in 5.30 s; residual branches off until 30% then ramped over 10%: 74.64% in 5.08 s (-0.43 pp, -4.1% time); off until 50%: 74.14% in 4.93 s (-0.92 pp, -6.9% time).
- **Interpretation:** R1 is disconfirmed: the recipe's own accuracy-time exchange rate (about 0.08 pp per 1% time, from S18's epoch steps) values the savings at -0.33 and -0.55 pp, less than the accuracy lost. As with freezing (F-009), short CIFAR-100 runs need full capacity throughout.
- **Decision consequence:** Reject progressive deepening; the representation portfolio X-003 is closed apart from dormant R7/R8, which their trigger conditions did not activate.
- **Resolution:** R1 rejected.

### F-028 — resolved, material

- **Observation:** S21 (10 seeds 1000-1009; control 75.22%): momentum 0.8 +0.23 pp; colour jitter +0.05 (+4.5% time); lookahead flush +0.03; BN recalibration +0.01; whitening from 960 images +0.00; bn_momentum 0.8 -0.04; random flip -0.09; momentum 0.9 -0.11; mixup -0.11; max+mean pool -0.15; SiLU -0.17; weight decay x2 -0.17; whitening bias 0.2 epochs -0.21; 3x3 whitening -0.28 (+11% time); CELU -0.43; cutout 8 -0.52; lookahead off -0.77; weight decay x0.5 -1.15. Combined SE is about 0.09 pp.
- **Interpretation:** No lineage or competitor lever beats the converged recipe except possibly momentum 0.8 (about 2.5 combined SE, one of 18 comparisons, so it needs fresh-seed confirmation). The lineage's augmentation additions (jitter, cutout, mixup), activations and whitening variants transfer negatively or neutrally to short single-view CIFAR-100 runs; lookahead remains essential.
- **Decision consequence:** Confirm and climb momentum on fresh seeds (S27); record every other S21 lever as tested and rejected or neutral.
- **Resolution:** Momentum candidate sent to S27.

### F-029 — resolved, contextual

- **Observation:** S27 (20 fresh seeds 1500-1519, 8.25 epochs): momentum 0.85 gives 75.16%, 0.8 gives 75.22%, 0.75 gives 75.22% (SE about 0.06 pp each).
- **Interpretation:** The S21 momentum-0.8 gain (+0.23 pp, 10 seeds, one of 18 comparisons) does not replicate: the fresh-seed difference is +0.06 pp, within one combined SE. It was a multiple-comparison artefact.
- **Decision consequence:** Keep momentum 0.85; no S21 lever enters the recipe.
- **Resolution:** Momentum unchanged.

### F-030 — resolved, material

- **Observation:** Modal A100 runs in the pinned stack (torch 2.4.0+cu124), 10 seeds 600-609, 8.25 epochs: on A100 80GB PCIe the converged recipe takes 6.169 s (m1) and 6.261 s (m2) per trial with default compile, 6.173 s with reduce-overhead and 6.115 s with max-autotune (181 s untimed build); 128/384/512 takes 5.76 s; prepare is 0.07 s; evaluation 0.10-0.12 s. Accuracy 75.15-75.20%. Paired with the local dev stack on the same seeds, A100 accuracy is -0.157 pp (SE 0.114).
- **Interpretation:** On the judging GPU the recipe trains about 9.6x faster than the organisers' 59.3 s baseline before the flatten-max pool and CUDA graphs. Compile-mode gains match the local proxy (-1.4% / -2.3%). Identical configs differ by about 1.5% between PCIe hosts, so timing comparisons must be paired on one host. The cross-stack accuracy gap is not significant at 10 seeds but would erase most of the 8.25-epoch margin if real.
- **Decision consequence:** Measure the current submission at 8.25/8.5/8.75 epochs on A100 with 40 fresh seeds plus a local twin (m3, s29) before fixing the budget.
- **Resolution:** Budget decision deferred to m3/s29.

### F-031 — resolved, material

- **Observation:** m3 (Modal, torch 2.4.0, 40 fresh seeds 1700-1739, current submission): 75.24% at 8.25 epochs (A100 SXM4, 5.24 s), 75.24% at 8.5 (SXM4, 5.73 s), 75.43% at 8.75 (A100 80GB PCIe, 6.45 s); untimed build 121-191 s; evaluation about 0.1 s. Its local twin s29 on the same seeds gives 75.27%, 75.31% and 75.46%. Paired A100 minus local: -0.03, -0.07 and -0.03 pp; pooled -0.043 pp (SE 0.035, n=120).
- **Interpretation:** Accuracy transfers between the dev stack and the pinned A100 stack to within 0.1 pp (R-007 satisfied); m1's -0.16 pp at 10 seeds was noise. The 8.25-epoch budget meets the 75.2% development criterion on the target stack. SXM hosts differ by up to 9% in time, so official-equivalent timing needs a PCIe host.
- **Decision consequence:** Keep 8.25 epochs (D-007 confirmed); run the official-equivalent 40-seed check on an A100 80GB PCIe host (T-015).
- **Resolution:** Cross-stack agreement established; budget confirmed.

### F-032 — resolved, material

- **Observation:** Official-equivalent runs of the submission defaults (40 fresh random uint32 seeds each, Modal, pinned torch 2.4.0 stack, 4 CPUs, network blocked): on an A100 80GB PCIe, 40/40 trials complete, mean 75.272% (sd 0.24 pp, worst 74.71%), prepare+train 6.027 s (sd 0.049 s), slowest evaluation 0.13 s, untimed build 192 s, 300 W limit; on an A100-SXM4-80GB, 75.253% in 5.545 s.
- **Interpretation:** The entry qualifies under official-equivalent conditions with an estimated probability of about 2e-7 that the official 40-trial mean falls below 75%; on the judging GPU model it trains 9.8x faster than the organisers' 59.3 s baseline. Runs used development mode (--submission-path) rather than --official; host-to-host timing varies by a few percent.
- **Decision consequence:** R-001 and R-005 have qualifying evidence; remaining work is the compliance review (T-016) and the team-lead-approved submission (T-017).
- **Resolution:** Qualification established.

### F-033 — open, material

- **Observation:** S30 (10 seeds 2000-2009, 8.25 epochs, lab equivalent of the submission, default compile): control 75.12%. Snapshot ensembles at 0.85 and 0.7+0.85 give 74.44% and 73.88%; a jointly trained 2x(96/256/448) ensemble 75.29% at +26% local time; stage-2 multi-exit (0.3/0.3) 74.24%; closed-form ridge head refit (lambda 1e-3, 1e-2) 74.14%; warmup-stable-decay from 0.6 gives 75.36%; cosine decay 74.98%; lookahead every 3 steps 75.08%; lookahead power 2 75.03%; class-balanced batch order 75.28% (SE about 0.07-0.09 pp per arm).
- **Interpretation:** Using the untimed evaluation budget does not help: snapshots from earlier in a short triangular run are much weaker than the final lookahead weights and dilute it, the joint ensemble costs more than its gain, the auxiliary exit and the refit head (fit to fast-weight features) both hurt. Two zero-cost candidates exceed two combined SE: WSD (+0.24 pp) and balanced order (+0.15 pp, about 1.4 SE); as the best two of 11 comparisons they need fresh-seed confirmation (S33) before adoption.
- **Decision consequence:** Reject snapshot, joint and multi-exit ensembles, head refit, cosine decay and lookahead cadence/power changes; confirm WSD (decay start 0.5/0.6/0.7) and balanced order on 20 fresh seeds in S33.

### F-034 — open, material

- **Observation:** M5 (Modal, two A100-SXM4-80GB hosts at 400 W and 500 W, 2 alternating-order blocks each, 5 seeds per arm-block, max-autotune): paired time change vs control within host-block: whitening-bias autograd off after freeze -3.02% (SE 0.25), loss inside the compiled graph -1.10% (0.29), cuDNN benchmark limit 0 -0.85% (0.27), coordinate-descent tuning -0.81% (0.19), skipping pool-discarded conv rows +1.19% (0.12), all five -3.37% (0.11); every arm consistent in sign across the 4 host-blocks; accuracies 75.12-75.24% (control 75.16%).
- **Interpretation:** H2-H5 are real exact savings on A100 SXM. H1 removes 8.8% of conv FLOPs at 32 px yet is slower: the explicit pad adds a copy and the unpadded odd shapes take slower cuDNN kernels, so FLOP counts do not predict conv time here. The all-lever arm carries the H1 penalty, so H2-H5 together should be near -4.5%; additivity and the PCIe magnitude are untested.
- **Decision consequence:** Reject H1. Time the H2+H5+H4+H3 stack against control on 4 hosts (M6), reporting PCIe hosts separately, then adopt the stack into the submission if consistent.

### F-035 — resolved, material

- **Observation:** M6 (Modal, 4 hosts: two A100 80GB PCIe 300 W, two A100-SXM4 400/500 W; 2 alternating-order blocks per host, 5 seeds per arm-block, max-autotune): on the PCIe hosts the control takes 6.120 s; whitening-bias autograd off after freeze -2.96% (SE 0.09, all 4 PCIe blocks -2.7 to -3.1%), plus loss in graph -3.14% (0.35), plus coordinate descent and cuDNN benchmark limit 0 -3.41% (0.38); on SXM -3.19%, -3.78%, -4.47%. Accuracies 75.20-75.30% (control 75.26%). Cold builds: control 120-244 s, full stack 259-409 s (limit 600 s).
- **Interpretation:** H2 is a robust exact saving of about 3% on the official card. H5, H3 and H4 help on the 400-500 W SXM hosts but add only about 0.2-0.5% on the power-capped PCIe card, within noise (PCIe blocks split 2-2 for H5), and H3+H4 raise the cold build to as much as 68% of the 600 s limit.
- **Decision consequence:** Adopt H2 into the submission (exact, CPU bit-identity with the lab holds); keep H5, H3 and H4 out (no significant PCIe gain; build-time risk). Re-qualify the updated submission on A100 PCIe before submitting.
- **Resolution:** H2 adopted in submissions/team_segal; H1, H3, H4, H5 not adopted.

### F-036 — open, material

- **Observation:** S31 (20 seeds 3000-3019, 8.25 epochs, control 75.15%, SE about 0.05 pp per arm): online LS 0.2 -0.10; PS-KD 0.3 +0.02, 0.6 -0.45; LS anneal to 0 -0.21; 1x1 head expansion 1024 -1.19; overlapping odd-map pools +0.04 (+4% time); ceil-mode stage 3 +0.23 (+5.5% time); cosine head s=20 -2.91; class-mean head init -0.17; fp32 master weights +0.03; clean tail 0.25/1.25 epochs +0.02/+0.02; conv wd x0.5 -0.74; head wd x0.5 +0.01; head lr x2/x0.5 -0.39/-0.18; bias scaler 32/128 +0.20/-0.28; init gain 0.5/0.7 +0.11/+0.08; gain 0.6 with shorter warm-up -0.01; momentum x0.5 at the switch +0.08; 20/32 blend +0.08; ETF head scale 1/6 and 1/3 -0.96/-0.51; LSE pool tau 0.5/1.0 +0.00/+0.17 (pp vs control).
- **Interpretation:** Soft-target, head-geometry and head-capacity levers do not help; fp16 update rounding is not limiting (master weights null); the clean tail is null. The weight-decay sensitivity lives in the scale-invariant convs (effective lr), not the head. Ceil-mode stage 3 gains less than its time cost. Three arms sit near the best-of-27 noise ceiling (about +0.15 pp): bias scaler 32, LSE pool tau 1.0, init gain 0.5.
- **Decision consequence:** Reject the rest; confirm bias scaler 32/16, LSE pool tau 1.0/2.0, init gain 0.5 and their stack on 20 fresh seeds (S37).

### F-037 — open, material

- **Observation:** S33 (20 fresh seeds 3300-3319): control 75.20%; WSD decay start 0.5/0.6/0.7: 75.26/75.18/75.13%; balanced order 75.29%; WSD 0.6 + balanced 75.40%. S34 (40 fresh seeds 3400-3439): control 75.12% (SE 0.037), WSD 0.6 + balanced 75.25% (SE 0.039); local time 5.37 vs 5.44 s.
- **Interpretation:** Neither lever confirms alone (WSD -0.02 to +0.06, balanced +0.09), but the combination replicates: +0.20 pp then +0.13 pp (2.4 SE) on independent fresh seeds, about +0.15 pp pooled over 60 seeds. Plausible mechanism: class-balanced batches reduce gradient noise, which pays off only when the lr stays at its peak through the 32 px switch. Worth about 0.15 epoch (about 1.8% time) if converted into a shorter budget.
- **Decision consequence:** Carry WSD 0.6 + balanced order as a candidate into a combined epoch-cut confirmation with any S37 survivors, timed on A100 PCIe, before adoption.

### F-038 — resolved, material

- **Observation:** S35 (20 seeds 3500-3519, equal step budget of 8.25 epochs, control 75.29%): dropping the easiest 20%/40% by banked loss from epoch 1: -0.05/-0.82 pp; easiest 30% from epoch 3: -0.38; hardest 5%/10%: -0.90/-1.73; 10% easiest + 10% hardest: -1.83; soft loss-proportional sampling of 70% (loss / gradient-norm score): -0.34/-0.28; easiest 30% by gradient-norm score: -0.34.
- **Interpretation:** No in-run pruning mode improves learning per step. The hardest examples are the most informative, not label noise: dropping only 5% of them costs 0.9 pp. Easy examples are nearly free to drop at 20% but give nothing back at equal steps, so there is no step count to save. In this underfit 8-epoch regime every image carries signal, which agrees with the earlier rejection of selection (F-010) and random retention.
- **Decision consequence:** Reject score-bank data pruning in all modes; close the data-selection dimension for this recipe.
- **Resolution:** Data pruning rejected.

### F-039 — open, material

- **Observation:** S37 (20 fresh seeds 3800-3819, current submission base with whitening-bias autograd off, control 75.16%): BN-bias lr scaler 32 75.35% (+0.19), scaler 16 75.50% (+0.34); LSE pool tau 1.0/2.0 75.27/75.15% with +4.0/+1.6% local time; conv init gain 0.5 75.22% (+0.06); stack scaler 32 + pool 1.0 + gain 0.5 75.40% (+0.24). Per-arm SE about 0.05-0.07 pp.
- **Interpretation:** The BN-bias learning rate is too high in the converged recipe: across S31 and S37 accuracy rises monotonically as the scaler falls (128: -0.28, 64: 0, 32: +0.19/+0.20, 16: +0.34), replicated on fresh seeds. The LSE pool does not pay for its time and init gain is within noise.
- **Decision consequence:** Scan the scaler further down (16, 8, 4, 2), test switch-time scaling and the stack with WSD + balanced order (S39); reject LSE pool and init gain; convert the best stack into an epoch cut.

### F-040 — open, material

- **Observation:** S36b (20 seeds 3600-3619, after fixing the momentum-buffer memory format; control 75.14% at 5.34 s local): narrowing at the 20->32 px switch to 128/384/512 (Taylor saliency) 74.88% at 5.09 s; to 448: Taylor 74.59%, random 74.62%, weight norm 74.58% (4.95-5.18 s); to 128/320/512 74.55% at 4.70 s; training 128/384/768 in the 20 px phase then 640: 75.29% at 5.41 s. S36 narrow-throughout controls: 512 -0.40 pp at -7.2%, 448 -0.65 pp at -9.9%.
- **Interpretation:** Gradient saliency adds nothing: random channel selection matches Taylor and weight-norm selection, so the transplant works by keeping width early, not by sniffing important channels. Narrowing trades about 0.05-0.06 pp per 1% local time, at the exchange rate (about 0.054), so no net gain. The reverse, extra width only in the cheap 20 px phase (768 then 640), gains +0.15 pp for about +1.4% local time, which is favourable if the time holds on A100.
- **Decision consequence:** Reject saliency narrowing; carry wide-early (768 -> 640) as a candidate for paired A100 timing and stacking with the BN-bias scaler.

### F-041 — resolved, material

- **Observation:** S38 (20 seeds 3900-3919; first two cells on the dev stack, the rest on Modal A100, torch 2.4): 128/384/640 at 8.25 epochs 75.16%; 128/384/512 at 8.75 / 9.25 epochs 74.97 / 75.26%; 128/384/448 at 9.0 / 9.5 epochs 74.79 / 75.07%; 128/320/512 at 9.0 epochs 74.84%.
- **Interpretation:** A narrower last stage needs about one extra epoch to match 640 at 8.25 (512 reaches 75.26% only at 9.25 epochs, +12% steps at about 7% cheaper steps, about +4% time), and 448 or 320/512 never catch up within the tested budgets. The width frontier has not moved: 640 at 8.25 epochs stays the efficient point.
- **Decision consequence:** Keep widths 128/384/640; close the width-reallocation direction.
- **Resolution:** Width unchanged.

### F-042 — resolved, material

- **Observation:** S40 (Modal A100, torch 2.4, 20 seeds 4200-4219): control 75.16%; post-add residual activation GELU(BN(conv3)+x) 75.03%; BN-bias scaler 16 75.56% (+0.40); scaler 16 with post-add activation 75.26%.
- **Interpretation:** The teammates' residual placement hurts on our recipe (-0.13 pp alone, -0.30 pp on the scaler-16 base). The BN-bias scaler 16 gain replicates for the fourth time and on the official torch 2.4 stack (+0.40 pp).
- **Decision consequence:** Reject post-add activation; scaler 16 proceeds to the epoch ladder (S41).
- **Resolution:** Post-add activation rejected.

### F-043 — resolved, material

- **Observation:** S41 (40 fresh seeds 4300-4339; first four cells on Modal A100 torch 2.4, last two on the dev stack): control 8.25 epochs 75.19% (SE 0.031); BN-bias scaler 16 at 8.25 / 8.0 / 7.75 epochs: 75.43 / 75.26 / 75.10% (SE 0.04); scaler 16 in the 20 px phase then 64: 8.0 / 7.75 epochs 75.17 / 75.00%. S39 (20 seeds): scaler 16/8/4/2 +0.30/+0.23/+0.07/-0.25 pp; 64->16 at the switch +0.08; 16->64 +0.37.
- **Interpretation:** Scaler 16 is a robust +0.24 to +0.40 pp across five sweeps and both stacks; it buys 0.25 epoch at matched accuracy (8.0 epochs: 75.26% vs 75.19% control), about -3% time. 7.75 epochs leaves only about 2.5 SE above 75%. The phase-switched variant does not replicate the S39 estimate.
- **Decision consequence:** Adopt bias_scaler 16 and 8.0 epochs in the submission; re-qualify on A100 PCIe with 40 fresh random seeds.
- **Resolution:** Adopted: bias_scaler 16, 8.0 epochs.

### F-044 — resolved, blocking

- **Observation:** M7a (Modal, NVIDIA A100 80GB PCIe 300 W, torch 2.4.0, 40 fresh random uint32 seeds, network blocked): updated submission defaults (whitening-bias autograd off, BN-bias scaler 16, 8.0 epochs) reach 75.290% mean single-view accuracy (SD 0.256) at 5.640 s mean prepare+train (SD 0.029). M7b landed on an A100-SXM4 (400 W): 75.332% at 5.255 s. T-015 measured 6.027 s for the previous defaults on a different PCIe host.
- **Interpretation:** The updated submission qualifies on the official card with a margin of about 7 SE (P(40-seed mean < 75%) negligible). Time is about 6% below the T-015 qualification, consistent with the paired -3% of the whitening change and the 3% step cut; the cross-host comparison carries 1.5-9% host variance.
- **Decision consequence:** The updated defaults are the current entry; any further adoption needs a fresh official-equivalent run.
- **Resolution:** Updated submission qualifies at 5.640 s on A100 PCIe.

### F-045 — resolved, material

- **Observation:** C2 (Modal, four A100-SXM4 hosts at 400/500 W, torch 2.4, 10 seeds per arm per host, arm order alternating between hosts): our submission 75.35% (SE 0.04) at 5.132 s; teammates' hypothesis-branch I001 (SGD, 8.5 epochs) 75.27% at +41.9% paired time (SE 0.22); I049 (Muon, 16/20/32 px, branch tip) 75.40% (SE 0.05) at +3.1% (0.41); I044 (Muon, 16/24/32 px) 75.66% (0.06) at +8.4% (0.31). The harness build timeout (600 s) failed 3 of 8 Muon runs (I044 on two hosts, I049 on one); all 4 runs of our submission and of I001 built.
- **Interpretation:** Our submission remains the fastest at matched accuracy on the same hosts. Their working Muon port is a credible alternative (I049 within about 3%, equal accuracy) but not better; I044's +0.31 pp costs 8.4% time, below our exchange rate (8.4% is worth about 0.45 pp). Their heavy build warm-up (50k synthetic images at every resolution, twice, plus a compiled Muon update) exceeds the 600 s build limit in about 40% of runs, which would fail an official evaluation.
- **Decision consequence:** Keep our submission as the team entry; if Muon is pursued, port their optimiser into the lab on our base and cap build time; tell the teammates their builds risk the 600 s timeout.
- **Resolution:** Our submission leads; Muon port optional.

### F-046 — resolved, material

- **Observation:** S42 (20 seeds 4500-4519, current submission base, control 75.27%): orthogonal init of the non-identity conv rows 75.27%; 2-D DCT filter-bank init 75.44% (+0.17); deterministic ZerO-style Hadamard init 75.23%; SkipInit residual gates from 0 / 0.25: 74.29 / 74.65%; DCT + gate 0 74.60%. S43 (40 fresh seeds 4600-4639): control 8.0 / 7.75 epochs 75.25 / 75.07%; DCT 8.0 / 7.75 epochs 75.36 / 75.23% (SE about 0.04).
- **Interpretation:** The DCT filter-bank start for each stage's widening outputs replicates (+0.17, +0.11, +0.15 pp) and buys 0.25 epoch: DCT at 7.75 epochs matches the 8.0-epoch control (75.23 vs 75.25%). Orthogonal and ZerO inits are neutral, so the gain comes from the frequency structure, not orthogonality. Gating residual branches toward identity costs 0.6-1.0 pp: the residual convs already start as identity and the short run needs the branches from the first step. All of these are unlearned constructions (the rule-compliant stand-in for build-time pre-training, which RULES.md section 3 forbids).
- **Decision consequence:** Adopt the DCT init with 7.75 epochs (372 steps); reject SkipInit; re-qualify on A100 PCIe.
- **Resolution:** Adopted: DCT init, 7.75 epochs.

### F-047 — resolved, blocking

- **Observation:** M8b (Modal, NVIDIA A100 80GB PCIe 300 W, torch 2.4.0, 40 fresh random seeds): submission defaults with the DCT init at 7.75 epochs reach 75.244% (SD 0.182, min 74.78) at 5.526 s mean prepare+train (SD 0.015). M8a on an A100-SXM4 400 W: 75.177% (SD 0.271) at 4.991 s.
- **Interpretation:** The entry still qualifies on the official card (margin above 5 SE on each block), now at 5.526 s: 2.0% below M7a and 8.3% below T-015, both cross-host comparisons.
- **Decision consequence:** The DCT/7.75-epoch defaults are the current entry; stage-1 freeze and depth cuts (S44) go to paired A100 timing next.
- **Resolution:** Entry qualifies at 5.526 s on A100 PCIe.

### F-048 — open, material

- **Observation:** S44 (20 seeds 4700-4719, bias scaler 16 at 8.0 epochs base, control 75.24% at 5.20 s local): stage-1 lr cooldown 0.6->0.8 75.19% (5.04 s); the same cooldown plus freezing stage 1 from 80% 75.19% (identical) at 4.72 s; cooldown 0.5->0.7 + freeze from 70% 74.99% (4.72 s); 18 px first phase 75.15%; depth-2 stage 1 at 8.25 epochs 75.07% (4.84 s); depth-2 stage 2 at 8.5 epochs 75.11% (4.85 s). Arms ran on two GPUs, so local times are indicative.
- **Interpretation:** Cooling stage 1's lr to zero by 80% costs about 0.05 pp, and the freeze that then skips stage 1's backward is exact (identical accuracy), saving an estimated 6-9% because it removes backward work in the expensive 32 px tail. Freezing from 70% costs 0.25 pp. Depth-2 stage cuts trade about 0.02 pp per 1% time, below the 0.054 exchange rate. The 18 px phase saves nothing measurable and costs accuracy.
- **Decision consequence:** Confirm freeze (80%, 75%), depth cuts and the freeze + depth stack on 40 fresh seeds on the current base (S45) and time them same-host on A100 (M9); reject the 18 px phase.

### F-049 — open, material

- **Observation:** S45 (40 fresh seeds 4800-4839, current base: DCT init, 7.75 epochs; control 75.20%): stage-1 cooldown + freeze from 80% 75.11%; from 75% 75.05%; depth-2 stage 1 at 8.0 epochs 74.97%; depth-2 stage 2 at 8.25 epochs 75.25%; freeze 80% + depth-2 stage 2 at 8.25 epochs 75.14%. S46 local paired timing (same arms on GPU 0 and reversed on GPU 1, 10 seeds each, max-autotune): -6.0%, -7.7%, -5.7%, -6.9% and -13.7% respectively, with the two GPUs within 1.3 points of each other.
- **Interpretation:** Removing stage 2's residual conv saves more time than the 0.5 extra epochs it needs: depth-2 stage 2 at 8.25 epochs is -6.9% at unchanged accuracy. The stage-1 freeze from 80% saves 6.0% for about -0.09 pp. Stacked they give -13.7% for -0.06 to -0.08 pp (60 seeds pooled), far better than the exchange rate, but the stack's 75.13% leaves a thin qualification margin. Depth-2 stage 1 only breaks even.
- **Decision consequence:** Price the stack's budget (8.25 / 8.5 epochs, freeze 80% / 85%) and depth-only at 8.0 / 8.25 on fresh seeds (S47) before adopting; reject depth-2 stage 1.

### F-050 — resolved, material

- **Observation:** S47 (40 fresh seeds 5100-5139, current base control 75.20%): depth-2 stage 2 at 8.0 / 8.25 epochs 75.03 / 75.21%; stack (stage-1 cooldown 0.6-0.8 + freeze 0.8 + depth-2 stage 2) at 8.25 / 8.5 epochs 75.07 / 75.20%; stack with freeze from 85% at 8.25 epochs 75.12%. S48 local paired timing (GPU 0 in order, GPU 1 reversed, 10 seeds each): stack at 8.5 epochs -9.2 / -12.2% (mean -10.7%), depth-2 stage 2 at 8.25 epochs -5.5 / -7.1% (mean -6.3%); accuracies 75.13 / 75.17 vs control 75.22% (20 seeds).
- **Interpretation:** The stack at 8.5 epochs matches control accuracy (about 75.18 vs 75.21% pooled over 60 seeds) at about 10.7% less time; at 8.25 epochs it saves more (-13.7%) but leaves only about 75.11%. Depth-2 stage 2 alone at 8.25 epochs is a safe -6.3% at equal accuracy.
- **Decision consequence:** Adopt the stack at 8.5 epochs (408 steps) as submission defaults (D-012).
- **Resolution:** Adopted: stage-1 freeze + depth-2 stage 2 at 8.5 epochs.

### F-051 — resolved, material

- **Observation:** S49 (local dev stack, the submission folder with its D-012 defaults, 40 fresh random uint32 seeds over GPUs 0 and 1): 75.129% (SD 0.243) and 75.180% (SD 0.279), pooled 75.154% (SE 0.041). With S47 (75.199%, 40 seeds) and S48 (75.13%, 20 seeds) the stack at 8.5 epochs averages about 75.17% over 100 seeds.
- **Interpretation:** The submission reproduces the lab result and clears 75% by about 3.8 SE on 40 seeds. Earlier A100 qualifications matched local means to within 0.03 pp (M7a vs S41, M8 vs S43), so the estimated risk that an official 40-seed mean falls below 75% is well under 1%, though thinner than at the previous defaults.
- **Decision consequence:** Keep D-012; an official-equivalent A100 PCIe run would confirm absolute time and margin if the user wants Modal used.
- **Resolution:** Submission defaults verified locally.

### F-052 — resolved, material

- **Observation:** S50 (20 seeds 5300-5319, D-012 base, control 75.18%): stage-2 freeze from 90% / 85% 74.98 / 74.87%; depth-2 stage 3 at 8.75 epochs 74.52%; BN-bias scaler 8 / 32 75.16 / 74.99%; lr 10 / 13 75.16 / 75.25%; warm-up peak 0.15 75.10%; switch at 45% / 55% 75.34 / 74.90%.
- **Interpretation:** The D-012 base is at a local optimum for these knobs; the switch point trades about 0.03-0.05 pp per 1% time, at or below the exchange rate; the stage-2 freeze and depth-2 stage 3 cost more accuracy than they save.
- **Decision consequence:** Keep the D-012 defaults; continue with S51 and S52.
- **Resolution:** No change.

### F-053 — resolved, material

- **Observation:** S51 (20 seeds 5400-5419, D-012 base, control 75.28%): BN-bias weight-decay multiplier 0.25 / 4 / 0: 74.96 / 75.19 / 74.84%; scaler 64 with bias decay x4: 75.16%; stage-1 cooldown+freeze 0.65-0.75 / 0.6-0.7 / 0.6-0.7 with stage-1 lr x1.3: 75.13 / 75.04 / 75.02%; identity + DCT perturbation of the square convs beta 0.25 / 0.5: 75.13 / 75.12%; identity + random perturbation beta 0.5: 75.25%; head-feature centring 74.41%; whitening eps 5e-3 / 5e-2: 75.23 / 75.19%.
- **Interpretation:** No learning-per-step lever improves the D-012 base. The bias decay arms separate size from noise: matching scaler 64's equilibrium bias size with less noise loses 0.32 pp while matching scaler 16's size with more noise loses only 0.12 pp, so the scaler-16 gain works mainly by keeping BN biases small; stronger decay adds nothing. Earlier stage-1 freezes lose 0.15-0.24 pp for 1.7-3.4% time (worse than the exchange rate). Square-conv perturbations, head centring and whitening eps are null or harmful.
- **Decision consequence:** Reject all S51 arms; keep D-012.
- **Resolution:** All rejected.

### F-054 — resolved, material

- **Observation:** S52 (20 seeds 5500-5519, D-012 base, control 75.22% at 4.53 s local on GPU 0): stage-1 update thinning every 2nd step over 0.6-0.8 / 0.5-0.8: 74.90 / 74.84%; every 3rd step over 0.5-0.8: 74.48%; stages 1+2 thinned over 0.6-0.8: 74.59%; weight-only freezes: stage-1 weights from 60% 74.84%, stage-3 weights from 90% 74.99%, both widening convs from 70% 74.26%, stages 2-3 weights from 90% 74.56%. Thinned arms ran 4.0-4.2 s (GPU 1, about 4% faster than GPU 0).
- **Interpretation:** Removing further gradient work beyond the exact stage-1 freeze costs more than it saves: thinning stage 1 trades about 0.07 pp per 1% time (worse than the 0.054 exchange rate), and freezing conv weights while BN biases train loses 0.2-1.0 pp. Unlike the exact freeze (which follows an lr cooldown to zero), these remove updates the network still uses.
- **Decision consequence:** Reject thinning and weight-only freezes; the D-012 structure stays.
- **Resolution:** All rejected.

### F-055 — resolved, material

- **Observation:** S53 (20 seeds 5600-5619, D-012 base, control 75.15% at 4.29 s local on GPU 1): native-scale 28 px window crops over 50-75% / 50-65% of training: 74.73 / 74.96% (the latter 4.17 s on the same GPU, -2.7%); 26 px over 50-75%: 74.48%; 28 px over 50-75% at 8.75 epochs: 74.86%.
- **Interpretation:** Window crops cost 0.2-0.7 pp and save far less time than their 22% MAC cut (about 2.7% for the 50-65% window): the 27/13/6 maps tile poorly, as with the skipped-rows conv. Below the exchange rate in every arm.
- **Decision consequence:** Reject window crops; the D-012 submission is converged against all round-3 agent hypotheses.
- **Resolution:** Rejected.

### F-056 — open, material

- **Observation:** S54 (20 seeds 5700-5719, D-012 base, control 75.06%): wide-early transplants (768 / 448 / all wide at 20 px, then 128/384/640) 75.16% each; LS 0.1 / 0.3 74.94 / 75.24%; momentum 0.8 / 0.9 75.15 / 74.99%; batch 768 75.43%. S55 (40 fresh seeds 5800-5839, control 75.11% at 4.306 s local): batch 768 at 8.5 / 8.0 / 7.75 epochs 75.37 / 75.15 / 74.90%; batch 896 at 8.25 epochs 75.14%; LS 0.3 75.27%; batch 768 at 8.0 epochs + LS 0.3 75.25%. Batch 768 at 8.0 epochs and batch 896 ran at -1.0% / -0.5% local time on the control's GPU.
- **Interpretation:** Smaller batches learn more per epoch but buy nothing at matched time (batch 768 at 8.0 epochs: +0.04 pp at -1%). Label smoothing 0.3 is now consistently better than 0.2 on the new base (+0.18, +0.16, +0.14 pp), unlike the flat 0.1-0.3 response on the old base. Wide-early gains (+0.10 pp) are within noise.
- **Decision consequence:** Reject batch-size changes and wide-early; price LS 0.3 as an epoch cut and test 0.35/0.4 (S58).

### F-057 — open, material

- **Observation:** S56 (20 seeds 5900-5919, D-012 base, control 75.12% at 4.33 s on GPU 1): stage-3 residual conv as 1x1 at 9.0 epochs 75.27% (4.39 s, GPU 0); stage-3 conv2 + residual as 1x1 at 9.5 / 8.5 epochs 74.47 / 74.02%; the same with a ceil-mode 4x4 terminal grid at 9.5 epochs 74.49%; with width 768 at 9.0 epochs 74.56%; stage 1 width 96 at 9.5 epochs 75.35% (4.57 s, GPU 0); centre-tap-only stage-3 square convs at 20 px, 8.75 epochs, 75.00% (4.33 s, GPU 1).
- **Interpretation:** Stage 3's 3x3 conv2 carries real spatial capacity on the 3x3 terminal grid (1x1 costs about 1 pp at equal epochs), but its residual conv works as 1x1 (+0.15 pp at about matched time). A thinner stage 1 at more epochs may sit on a better frontier (+0.23 pp). Centre-tap pruning at 20 px loses at equal time.
- **Decision consequence:** Confirm the 1x1 stage-3 residual and stage-1 width 96 (and their combination) with 40 fresh seeds and same-GPU paired timing (S59); reject the other topology arms.

### F-058 — resolved, material

- **Observation:** S57 (local paired timing, previous vs patched submission, 10 seeds per GPU, GPU 0 prev->new and GPU 1 new->prev): total 4.433 -> 4.378 s (-1.24%) and 4.271 -> 4.178 s (-2.17%); prepare 65 -> 39 ms on both GPUs; accuracy identical per GPU (75.23 / 75.14%).
- **Interpretation:** The round-4 audit's exact fixes save about 1.7% with bit-identical results: the vectorised identity init removes 26 ms of per-element GPU copies from prepare, the gather crop removes host syncs and the full-epoch flip copy, and stopping at the last lookahead update drops three steps whose updates the final copy discards.
- **Decision consequence:** Adopt all three fixes and the full-size warm-up in the submission (exact; no accuracy run needed).
- **Resolution:** Adopted (exact).

### F-059 — open, material

- **Observation:** S58 (40 fresh seeds 6100-6139, D-012 base with trimmed tail, control 8.5 epochs 75.12%): label smoothing 0.3 at 8.5 epochs 75.26%; at 8.25 epochs LS 0.3 / 0.35 / 0.4: 75.13 / 75.16 / 75.21%.
- **Interpretation:** Stronger label smoothing is worth about 0.25 epoch on the new base: LS 0.3-0.4 at 8.25 epochs matches or beats the 8.5-epoch control, a 3% step cut, and accuracy still rises from 0.3 to 0.4.
- **Decision consequence:** Extend the trend (0.5) and test 8.0 epochs (S61) before folding the best smoothing and budget into a combined final configuration with the topology survivors.

### F-060 — resolved, material

- **Observation:** S60 (local paired timing, max-autotune, GPU 0 in order and GPU 1 reversed, 10 seeds each) vs control: stage-3 residual 1x1 at 9.0 epochs +19.8% / +0.6%; stage-1 width 96 at 9.5 epochs +23.1% / +18.0%; both at 10.0 epochs +3.9% / -0.4%. S59 (40 fresh seeds 6200-6239): control 75.22%; 75.26%, 75.34% and 75.39% respectively. No GPU contention was logged.
- **Interpretation:** The predicted per-step savings of the 1x1 residual and thinner stage 1 did not materialise under max-autotune (narrow 96-channel and 1x1 shapes are not cheaper in practice), so the extra epochs needed to match accuracy cost more time than they save; the accuracy gains (+0.04 to +0.17 pp) do not pay for the time. The combined arm is near break-even but its timing is inconsistent across GPUs.
- **Decision consequence:** Reject the topology changes; the D-012 structure stays.
- **Resolution:** Rejected.

### F-061 — resolved, material

- **Observation:** S61 (40 fresh seeds 6400-6439, D-012 base with trimmed tail, control 8.5 epochs 75.17%): label smoothing 0.4 / 0.5 at 8.25 epochs 75.17 / 75.05%; at 8.0 epochs 74.97 / 74.90%. Pooled with S58, LS 0.4 at 8.25 epochs averages 75.19% over 80 seeds against 75.15% for the matched 8.5-epoch controls.
- **Interpretation:** Label smoothing 0.4 buys a quarter epoch at matched accuracy (395 instead of 405 trained steps, about -2.5%); 0.5 is past the optimum and 8.0 epochs is too short.
- **Decision consequence:** Adopt label smoothing 0.4 at 8.25 epochs (D-013) and verify the submission folder on 40 fresh seeds (S62).
- **Resolution:** Adopted: LS 0.4, 8.25 epochs.

### F-062 — resolved, material

- **Observation:** S62 (local dev stack, the submission folder with D-013 defaults and the exact timed-path fixes, 40 fresh random uint32 seeds over GPUs 0 and 1): 75.195% (SD 0.236, min 74.80) at 4.160 s and 75.160% (SD 0.227, min 74.88) at 4.068 s; pooled 75.177% (SE 0.036); prepare 39 ms. S49 measured 4.408 / 4.243 s for the previous defaults.
- **Interpretation:** The submission reproduces the lab result (about 75.18%) and is about 4-6% faster locally than at S49, consistent with the exact fixes (-1.7%) plus 2.5% fewer steps; margin above 75% is about 5 SE.
- **Decision consequence:** Commit D-013 as the current entry; an official A100 PCIe run would confirm absolute time if the user wants Modal used.
- **Resolution:** Submission verified locally.

### B-001 — external, resolved

- **Issue:** No A100 80GB PCIe is available: the local GPUs are Blackwell (sm_120), which the pinned torch 2.4.0 cannot run, and renting an A100 requires team-lead approval of provider and budget.
- **Consequence:** A100 timing (R-005) and cross-stack accuracy agreement (R-007) cannot be measured; the regime decision T-006 waits on them, while the local frontier probe T-005 can proceed.
- **Resolution:** A100 access via Modal (workspace gereonelvers99, team-lead supplied token): m1 ran the pinned torch 2.4.0 stack on A100 80GB PCIe and A100-SXM4-80GB GPUs.

### B-002 — external, resolved

- **Issue:** No A100 80GB PCIe is available: the local GPUs are Blackwell (sm_120), which the pinned torch 2.4.0 cannot run, and renting an A100 requires team-lead approval of provider and budget.
- **Consequence:** A100 timing (R-005) and cross-stack accuracy agreement (R-007) cannot be measured; the regime decision T-006 waits on them, while the local frontier probe T-005 can proceed.
- **Resolution:** A100 access via Modal (workspace gereonelvers99, team-lead supplied token): m1 ran the pinned torch 2.4.0 stack on A100 80GB PCIe and A100-SXM4-80GB GPUs.

## Decisions and changes

### D-001 — provisional

- **Question:** How should exploration estimate training time while no A100 80GB PCIe is available (B-001, B-002)?
- **Decision:** Per the team lead's directive, explore on one local GPU (GPU 1, one slot) and rank configurations by a local proxy: eager harness prepare+train time on the otherwise idle GPU, alongside single-view dev-stack accuracy. Every proxy-based selection is provisional and reopens when A100 per-epoch timings exist.
- **Rationale:** The A100 is blocked on an external approval; local accuracy is the dominant uncertainty and is hardware-independent to first order, while relative per-epoch cost on an idle Blackwell GPU is the best available ordering of candidate shapes.
- **Alternatives:** Wait for the A100 before any selection: rejected because it stalls the accuracy frontier, which does not need the A100.; Rank by analytic FLOPs only: rejected as the primary proxy because the airbench paper reports FLOP cuts that did not cut A100 wall time; FLOPs are recorded as a secondary check.

### D-002 — provisional

- **Question:** Which shape, augmentation and budget is the base for lever comparisons?
- **Decision:** 128/384/512 with three convs per block and a residual, translate 2, about 9 epochs (8.28 s proxy to 75.3%).
- **Rationale:** Lowest interpolated proxy time in S3; translate 2 beats 4 by about 0.4-0.5 pp at fixed epochs.
- **Alternatives:** Translate 4 (P1 default for the shape): 9.02 s, rejected.; 128/512/768: fewer epochs (7.1) but 9.43 s, rejected for now and retested at translate 2 in S4.; Two-conv 2x width: 12.0 s in P1, rejected.

### D-003 — provisional

- **Question:** Which S4 levers enter the climbing recipe?
- **Decision:** Adopt progressive resizing (24 px then 32 px); keep SGD with lookahead, batch 1024, label smoothing 0.2 and translate 2.
- **Rationale:** Only resizing lowers proxy time beyond noise (-15%); other levers tie or lose accuracy at fixed epochs.
- **Alternatives:** Muon (as ported): rejected, -0.8 to -1.4 pp and slower steps.; Batch 1536/2048: rejected, slower to target.; Translate 1 / label smoothing 0.1: equivalent; keep defaults to limit changes.

### D-004 — provisional

- **Question:** Should the recipe compile its training step, and how?
- **Decision:** Compile the training model in build with torch.compile(mode='default', dynamic=False) and use fused SGD; evaluation keeps the eager module.
- **Rationale:** About 12% lower time at unchanged accuracy over 10 fresh seeds; build cost stays far below the 600 s limit; dynamic=False keeps one specialised graph per resolution.
- **Alternatives:** Eager: rejected, 12% slower.; max-autotune (CUDA graphs): not yet re-measured with dynamic=False; S7 showed 13% with the dynamic-shape bug, so it stays a candidate for the A100.

### D-005 — provisional

- **Question:** Which widths form the base after S12?
- **Decision:** 128/384/640, three convs per block, translate 2, 20 px then 32 px at half-way, about 8.7 epochs.
- **Rationale:** Lowest proxy time to 75.3% in S12 (6.29 s, about 6% under the 128/384/512 base) and more accuracy margin per epoch.
- **Alternatives:** 128/384/768: 6.52 s, rejected.; 128/320/768: 6.45 s, within noise but more total width; kept as runner-up.; 128/384/512: does not reach 75.3% by 9 epochs on fresh seeds; superseded.

### D-006 — provisional

- **Question:** What logit scale should the head use?
- **Decision:** scaling_factor 1/6 (1.5x the airbench 1/9).
- **Rationale:** +0.29-0.34 pp at fixed epochs at zero cost over 10 fresh seeds, replicated by the competitor's independent run.
- **Alternatives:** 1/9 (airbench default): superseded.; 2/9: worse than 1/6.

### D-007 — provisional

- **Question:** What training budget does the converged recipe use?
- **Decision:** 8.25 epochs (40-seed mean 75.26%, 5.38 s local compiled proxy).
- **Rationale:** Shortest budget whose 40-seed fresh mean meets the 75.2% criterion (F-005); official-mean risk is negligible on the dev stack.
- **Alternatives:** 8.5 epochs (75.33%, +2.6% time): fallback if A100 accuracy is lower.; 8.75 epochs (75.43%, +5.8% time): conservative fallback.; 8.0 epochs (75.02%): rejected, below the criterion.

### D-008 — settled

- **Question:** Which base regime does the entry use, given the frontier and A100 timings?
- **Decision:** 128/384/640 airbench96-style net (three convs per block with a residual), translate 2, 20 px then 32 px at half-way, logit scale 1/6, max-autotune compile with flatten-max pooling, 8.25 epochs.
- **Rationale:** Lowest local proxy time to the target across the frontier and climb (F-003, F-014, F-017, F-021); on the A100 stack it reaches 75.24% over 40 fresh seeds at 8.25 epochs (F-031) and about 6.1 s per trial on A100 80GB PCIe (F-030); accuracy transfers between stacks within 0.1 pp.
- **Alternatives:** 128/384/512: 74.67% at 8.25 epochs on A100 PCIe (5.76 s), below the target; rejected.; 128/384/768: 75.59% at 8.25 epochs on A100 SXM4; reaches the target in fewer epochs but costs more per epoch; same-host timing would be needed to beat 640 (reopening condition).; Two-conv 2x width and ResNet-9: slower to target (F-003, F-004).

### D-009 — settled

- **Question:** Which exact systems levers enter the submission after M5 and M6?
- **Decision:** Adopt whitening-bias autograd removal after the freeze (H2); do not adopt loss-in-graph (H5), coordinate-descent tuning (H4), cuDNN benchmark limit 0 (H3) or skipping pool-discarded rows (H1).
- **Rationale:** H2 saves 2.96% (SE 0.09) on the official A100 PCIe and is exact; H5/H3/H4 gains on PCIe are within noise and H3+H4 push the cold build to 409 s of 600 s; H1 is 1.2% slower.
- **Alternatives:** Adopt the full four-lever stack (-3.41% on PCIe, not significantly better than H2 alone, larger build-time risk); Adopt nothing until a PCIe-only timing run (unnecessary: H2 is exact and consistent on every host-block)

### D-010 — settled

- **Question:** Which accuracy lever and budget enter the submission after S37-S41?
- **Decision:** Set bias_scaler 16 and epochs 8.0 (384 steps); keep everything else.
- **Rationale:** Scaler 16 replicated in five sweeps (+0.24 to +0.40 pp) and holds control accuracy at 8.0 epochs (75.26% vs 75.19%, 40 seeds), a 3% step cut with an unchanged qualification margin; 7.75 epochs is too close to 75%.
- **Alternatives:** 7.75 epochs (75.10%, about 2.5 SE above the threshold; qualification risk too high); WSD + balanced order on top (no added gain over scaler 16 in S39)

### D-011 — settled

- **Question:** Does the DCT filter-bank initialisation enter the submission, and at what budget?
- **Decision:** Adopt the DCT init for the non-identity rows of each stage's widening conv and cut the budget to 7.75 epochs (372 steps).
- **Rationale:** Replicated in S42 and S43 (+0.11 to +0.17 pp); at 7.75 epochs it matches the 8.0-epoch control on 40 fresh seeds, a 3% step cut at no time cost; bit-identical to the lab on CPU.
- **Alternatives:** Keep 8.0 epochs and bank the accuracy margin (+0.11 pp); 7.5 epochs (untested; margin would likely fall below 75%)

### D-012 — settled

- **Question:** Which budget-reallocation changes enter the submission after S44-S48?
- **Decision:** Stage 2 without its residual conv, stage-1 lr cooldown from 60% to 80% with stage 1 frozen (no autograd) from 80%, 8.5 epochs (408 steps).
- **Rationale:** About -10.7% local paired time (S48; -13.7% at 8.25 epochs in S46 plus 3% for the extra quarter epoch) at control accuracy (75.20% vs 75.20% in S47, 40 fresh seeds); the freeze is exact and both changes are bit-identical to the lab.
- **Alternatives:** Depth-2 stage 2 alone at 8.25 epochs (-6.3%, equal accuracy; safer but slower); Stack at 8.25 epochs (-13.7% but about 75.11%, too little qualification margin)

### D-013 — settled

- **Question:** Which label smoothing and budget does the submission use after S54-S61?
- **Decision:** Label smoothing 0.4 with 8.25 epochs (396-step schedule, 395 trained steps).
- **Rationale:** LS 0.3-0.4 is consistently better than 0.2 on the new structure (S54, S55, S58); LS 0.4 at 8.25 epochs matches the 8.5-epoch control over 80 seeds (75.19 vs 75.15%), about 2.5% fewer steps; 0.5 and 8.0 epochs fall short.
- **Alternatives:** LS 0.3 at 8.25 epochs (75.13%, slightly less margin); Keep LS 0.2 at 8.5 epochs (no time gain)

## Deferred or rejected work

- **T-007: Add-on levers at the selected base (P2)** — Superseded by T-013: probes S4-S15 tested Muon, resizing, selection, batch size, regularisation, compile and further levers at the climbed base (F-007 to F-019); its premise of a separate add-on pass after an A100 regime decision no longer holds because the climb ran on the local proxy under D-001.
- **T-008: Adopt or reject add-on levers** — Superseded by T-013 decisions D-003 to D-007, which record adopted and rejected levers with evidence and reopening conditions.
- **T-009: Converge and simplify the final submission** — Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance.
- **T-010: Official-equivalent 40-seed qualification on the A100 PCIe** — Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance.
- **T-011: Fresh-context compliance review and independence checks** — Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance.
- **T-012: Open the upstream pull request after approval** — Dependency chain through cancelled T-008 no longer reflects the programme; reissued against T-013 as a successor task with the same objective and acceptance.

## Review and assurance

- No assessment requested.

## Programme detail

### Outcomes

| ID | Outcome | Derived state |
| --- | --- | --- |
| O-001 | A qualifying, compliant submission is demonstrated under official-equivalent conditions. | unresolved |
| O-002 | The submitted recipe is the fastest qualifying configuration the evidence supports. | supported |
| O-003 | Exploration is reproducible, resumable and scales across available GPUs and agents. | supported |
| O-004 | The submission reaches the organisers in the required form with team-lead approval. | unresolved |

### Requirements

| ID | Criticality | Requirement | Derived state |
| --- | --- | --- | --- |
| R-001 | core | In the pinned container on an A100 80GB PCIe with the official launch flags, the entry completes 40 fresh trials, every trial succeeds within the 600 s training and 5 s evaluation limits, and the mean top-1 exceeds 75% by a margin that keeps the estimated official qualification risk at or below 1% given the measured per-trial standard deviation (75.2% when that deviation is at most 0.30 pp). | supported |
| R-002 | core | The entry satisfies RULES.md sections 1 to 3: no real data, seeds or data-derived constants in import or build; no learned state carried across trials; single-view evaluation without state change; no test-set use; no measurement interference; only pinned dependencies. | pending_evidence |
| R-003 | core | The base regime (architecture, width, depth, batch size, epochs) is selected as the minimum interpolated A100 PCIe time at which the single-view 5-seed mean reaches 75.3%, over a measured frontier of at least four widths and four epoch counts. | supported |
| R-004 | core | Each add-on lever (optimiser, resolution schedule, example selection, regularisation) is adopted only if it lowers time at matched accuracy beyond seed noise over at least 10 seeds at the selected base; every rejected lever has a recorded reason. | supported |
| R-005 | core | A100 80GB PCIe per-epoch and fixed preparation times are measured in the pinned container for every frontier candidate, and the final recipe's 40-trial mean time is recorded with its standard deviation and GPU telemetry. | supported |
| R-006 | supporting | Any configuration in the exploration space runs from a declarative sweep file on any free local GPU through the official harness, resumes after interruption without repeating completed runs, and its collated tables are reproducible from raw results. | supported |
| R-007 | supporting | Single-view accuracy measured on the local development stack agrees with the pinned A100 stack within two standard errors for a reference configuration, or the discrepancy is quantified and applied as a correction. | supported |
| R-008 | core | The pull request to the upstream repository adds only submissions/<team>/ with its source and README, runs its final recipe from default settings, contains no weights, data or results, and was approved by the team lead before opening. | pending_evidence |

### Worksets

| Sequence | Workset | Decision boundary | Status |
| ---: | --- | --- | --- |
| 1 | Exploration enablement | Can the team run the exploration grid at scale and trust its outputs? | complete |
| 2 | Width by epochs frontier | Is the competition capacity-bound or throughput-bound, and which base regime wins? | active |
| 3 | A100 PCIe calibration | Which time model and which accuracy correction apply to frontier decisions? | complete |
| 4 | Add-on levers at the selected base | Which levers enter the final recipe? | planned |
| 5 | Convergence and assurance | Is the exact candidate qualifying, compliant and ready to submit? | active |
| 6 | Submission | Has the approved candidate been submitted in the required form? | planned |

### Explorations

- **X-001:** Where does the single-view width by epochs frontier cross 75% on CIFAR-100, and is the minimum-time regime capacity-bound or throughput-bound? (closed; test-ready=true)
- **X-002:** Which levers lower local proxy time to a 75.3% single-view mean from the airbench96-shape base, and when does the climb saturate? (saturated; test-ready=true)
- **X-003:** Do representation or learning approaches outside the airbench convnet-SGD-CE family lower time to a 75% single-view CIFAR-100 mean? (closed; test-ready=true)
- **X-004:** What are A100 80GB PCIe timings for the frontier candidates, and does dev-stack accuracy transfer to the pinned A100 stack? (closed; test-ready=true)
- **X-005:** Do structural, learning-dynamics or A100 systems levers outside the climbed parameter space lower A100 PCIe time to a 75.2% single-view mean beyond seed and host noise? (saturated; test-ready=true)

### Exact frontier

- State: **needs_design**
- Active: T-019
- Eligible: none
- Unresolved outcomes: O-001, O-002, O-003, O-004
- Unresolved requirements: R-001, R-002, R-003, R-004, R-005, R-006, R-007, R-008
- Pending assessments: none
- Human engagement: none
<!-- writing-tools:generated:end -->
