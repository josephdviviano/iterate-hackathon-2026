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

- **State:** continue
- **Mission:** Submit a rule-compliant CIFAR-100 speedrun entry whose official 40-seed evaluation on one NVIDIA A100 80GB PCIe qualifies (mean top-1 at least 75%) at the lowest mean prepare+train time the team can demonstrate, with recorded evidence for every adopted and rejected technique.
- **Root question:** Which compliant recipe minimises mean A100 PCIe prepare+train time while keeping the official 40-seed mean top-1 at or above 75% with a qualification risk of about 1% or less?
- **Why:** Executable work remains. Select among eligible tasks by consequence and decision value, never by identifier.
- **Next:** Continue T-013 (Local-proxy hill climb to convergence (X-002)).
- **Open human feedback:** none

## Work completed and underway

| Task | Type | State | Outcome or purpose |
| --- | --- | --- | --- |
| [T-001](tasks/T-001-local-accuracy-stack-on-the-blackwell-gpus.md) — Local accuracy stack on the Blackwell GPUs | delivery | completed | Dev stack torch 2.7.1+cu128 runs the official harness on real data on both local GPUs; setup script and freeze reproduce it from a fresh worktree. |
| [T-002](tasks/T-002-parametrised-airbench-lineage-recipe-substrate-in-the-team-folde.md) — Parametrised airbench-lineage recipe substrate in the team folder | delivery | completed | Parametrised recipe passes CPU contract tests (reset, invalid params, defaults, harness smoke for three variants) on the pinned stack and trains on real data (default 69.5% single-view, eager evaluation). |
| [T-003](tasks/T-003-declarative-multi-gpu-sweep-runner-collation-and-runbook.md) — Declarative multi-GPU sweep runner, collation and runbook | delivery | completed | Sweep runner dispatches across both GPUs through the harness, resumes only interrupted configs, records failures, and collates reproducibly; runbook replayed from a fresh worktree. |
| [T-004](tasks/T-004-a100-pcie-timing-calibration-and-cross-stack-accuracy-agreement.md) — A100 PCIe timing calibration and cross-stack accuracy agreement | exploration | blocked | On a rented A100 80GB PCIe in the pinned container, measure per-epoch and fixed preparation time for each frontier width with telemetry, re-time a ResNet-9 reimplementation against the 59.3 s baseline, and compare a reference configuration's 10-seed accuracy with the dev stack. |
| [T-005](tasks/T-005-single-view-width-by-epochs-frontier-p1.md) — Single-view width by epochs frontier (P1) | exploration | completed | Frontier measured over 4 shapes x 4 epoch counts x 5 seeds plus a ResNet-9 arm; airbench96 shape reaches 75.37% at 10 epochs, 2x crosses at 13.7 epochs; capacity-bound (F-003). |
| [T-006](tasks/T-006-select-the-base-regime.md) — Select the base regime | decision | proposed | Choose architecture, width, depth, batch size and epoch count by minimum interpolated A100 PCIe time at a 75.3% single-view mean, steelmanning the runner-up regime. |
| [T-007](tasks/T-007-add-on-levers-at-the-selected-base-p2.md) — Add-on levers at the selected base (P2) | exploration | proposed | Compare Muon versus SGD with lookahead, progressive resizing, in-run proxy-loss example selection and label-smoothing level at matched accuracy with at least 10 seeds per arm at the selected base. |
| [T-008](tasks/T-008-adopt-or-reject-add-on-levers.md) — Adopt or reject add-on levers | decision | proposed | Record which levers enter the final recipe and why each rejected lever was rejected. |
| [T-009](tasks/T-009-converge-and-simplify-the-final-submission.md) — Converge and simplify the final submission | delivery | proposed | Remove exploration-only parameters and code paths, make the defaults run the selected recipe, write the submission README, then run one simplify-codebase pass over the integrated team folder. |
| [T-010](tasks/T-010-official-equivalent-40-seed-qualification-on-the-a100-pcie.md) — Official-equivalent 40-seed qualification on the A100 PCIe | assurance | proposed | Run the frozen candidate in the pinned container on an A100 80GB PCIe with a private 40-seed file, cpus 4 and network none; record results, telemetry and the R-001 risk calculation. |
| [T-011](tasks/T-011-fresh-context-compliance-review-and-independence-checks.md) — Fresh-context compliance review and independence checks | assurance | proposed | Review the frozen candidate source against every RULES.md section 3 bullet without implementation narrative, and run repeat-seed, reordered-seed and fresh-process independence checks. |
| [T-012](tasks/T-012-open-the-upstream-pull-request-after-approval.md) — Open the upstream pull request after approval | delivery | proposed | After a recorded team-lead approval, create a clean branch from upstream main containing only the team folder and open the pull request. |
| [T-013](tasks/T-013-local-proxy-hill-climb-to-convergence-x-002.md) — Local-proxy hill climb to convergence (X-002) | exploration | in_progress | From the P1 frontier, climb one lever at a time (width and stage allocation, epochs, optimiser, batch size, resolution schedule, regularisation, learning rates) on GPU 1, keeping a change only when the interpolated proxy time to a 75.3% single-view mean falls beyond seed noise; stop when no accessible lever improves it. |

## Issues identified

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

### B-001 — external, open

- **Issue:** No A100 80GB PCIe is available: the local GPUs are Blackwell (sm_120), which the pinned torch 2.4.0 cannot run, and renting an A100 requires team-lead approval of provider and budget.
- **Consequence:** A100 timing (R-005) and cross-stack accuracy agreement (R-007) cannot be measured; the regime decision T-006 waits on them, while the local frontier probe T-005 can proceed.

### B-002 — external, open

- **Issue:** No A100 80GB PCIe is available: the local GPUs are Blackwell (sm_120), which the pinned torch 2.4.0 cannot run, and renting an A100 requires team-lead approval of provider and budget.
- **Consequence:** A100 timing (R-005) and cross-stack accuracy agreement (R-007) cannot be measured; the regime decision T-006 waits on them, while the local frontier probe T-005 can proceed.

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

## Deferred or rejected work

- None recorded.

## Review and assurance

- No assessment requested.

## Programme detail

### Outcomes

| ID | Outcome | Derived state |
| --- | --- | --- |
| O-001 | A qualifying, compliant submission is demonstrated under official-equivalent conditions. | unresolved |
| O-002 | The submitted recipe is the fastest qualifying configuration the evidence supports. | unresolved |
| O-003 | Exploration is reproducible, resumable and scales across available GPUs and agents. | unresolved |
| O-004 | The submission reaches the organisers in the required form with team-lead approval. | unresolved |

### Requirements

| ID | Criticality | Requirement | Derived state |
| --- | --- | --- | --- |
| R-001 | core | In the pinned container on an A100 80GB PCIe with the official launch flags, the entry completes 40 fresh trials, every trial succeeds within the 600 s training and 5 s evaluation limits, and the mean top-1 exceeds 75% by a margin that keeps the estimated official qualification risk at or below 1% given the measured per-trial standard deviation (75.2% when that deviation is at most 0.30 pp). | pending_evidence |
| R-002 | core | The entry satisfies RULES.md sections 1 to 3: no real data, seeds or data-derived constants in import or build; no learned state carried across trials; single-view evaluation without state change; no test-set use; no measurement interference; only pinned dependencies. | pending_evidence |
| R-003 | core | The base regime (architecture, width, depth, batch size, epochs) is selected as the minimum interpolated A100 PCIe time at which the single-view 5-seed mean reaches 75.3%, over a measured frontier of at least four widths and four epoch counts. | pending_evidence |
| R-004 | core | Each add-on lever (optimiser, resolution schedule, example selection, regularisation) is adopted only if it lowers time at matched accuracy beyond seed noise over at least 10 seeds at the selected base; every rejected lever has a recorded reason. | pending_evidence |
| R-005 | core | A100 80GB PCIe per-epoch and fixed preparation times are measured in the pinned container for every frontier candidate, and the final recipe's 40-trial mean time is recorded with its standard deviation and GPU telemetry. | blocked |
| R-006 | supporting | Any configuration in the exploration space runs from a declarative sweep file on any free local GPU through the official harness, resumes after interruption without repeating completed runs, and its collated tables are reproducible from raw results. | supported |
| R-007 | supporting | Single-view accuracy measured on the local development stack agrees with the pinned A100 stack within two standard errors for a reference configuration, or the discrepancy is quantified and applied as a correction. | blocked |
| R-008 | core | The pull request to the upstream repository adds only submissions/<team>/ with its source and README, runs its final recipe from default settings, contains no weights, data or results, and was approved by the team lead before opening. | pending_evidence |

### Worksets

| Sequence | Workset | Decision boundary | Status |
| ---: | --- | --- | --- |
| 1 | Exploration enablement | Can the team run the exploration grid at scale and trust its outputs? | complete |
| 2 | Width by epochs frontier | Is the competition capacity-bound or throughput-bound, and which base regime wins? | active |
| 3 | A100 PCIe calibration | Which time model and which accuracy correction apply to frontier decisions? | active |
| 4 | Add-on levers at the selected base | Which levers enter the final recipe? | planned |
| 5 | Convergence and assurance | Is the exact candidate qualifying, compliant and ready to submit? | planned |
| 6 | Submission | Has the approved candidate been submitted in the required form? | planned |

### Explorations

- **X-001:** Where does the single-view width by epochs frontier cross 75% on CIFAR-100, and is the minimum-time regime capacity-bound or throughput-bound? (closed; test-ready=true)
- **X-002:** Which levers lower local proxy time to a 75.3% single-view mean from the airbench96-shape base, and when does the climb saturate? (test_ready; test-ready=true)

### Exact frontier

- State: **continue**
- Active: T-013
- Eligible: none
- Unresolved outcomes: O-001, O-002, O-003, O-004
- Unresolved requirements: R-001, R-002, R-003, R-004, R-005, R-007, R-008
- Pending assessments: none
- Human engagement: none
<!-- writing-tools:generated:end -->
