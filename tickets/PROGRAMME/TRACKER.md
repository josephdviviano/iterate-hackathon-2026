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
- **Next:** Continue T-001 (Local accuracy stack on the Blackwell GPUs).
- **Open human feedback:** none

## Work completed and underway

| Task | Type | State | Outcome or purpose |
| --- | --- | --- | --- |
| [T-001](tasks/T-001-local-accuracy-stack-on-the-blackwell-gpus.md) — Local accuracy stack on the Blackwell GPUs | delivery | in_progress | Provide a reproducible dev stack (torch 2.7.1+cu128) beside the pinned stack, with CIFAR-100 downloaded, so the official harness runs real-data trials on both local GPUs. |
| [T-002](tasks/T-002-parametrised-airbench-lineage-recipe-substrate-in-the-team-folde.md) — Parametrised airbench-lineage recipe substrate in the team folder | delivery | ready | Implement build/prepare/train for an airbench-lineage CIFAR-100 recipe whose architecture, width, depth, epochs, batch size, optimiser, schedule and augmentation are selected by declarative parameters, with compliance invariants (synthetic-only build, in-place per-trial reset, single-view stateless forward). |
| [T-003](tasks/T-003-declarative-multi-gpu-sweep-runner-collation-and-runbook.md) — Declarative multi-GPU sweep runner, collation and runbook | delivery | ready | Run sweep files of configurations and seeds through the official harness on free local GPU slots, skip completed configurations on resume, record failures, and collate mean, sd and n per configuration into reproducible tables; document setup, sweeps, resume and handoff in an executable runbook. |
| [T-004](tasks/T-004-a100-pcie-timing-calibration-and-cross-stack-accuracy-agreement.md) — A100 PCIe timing calibration and cross-stack accuracy agreement | exploration | ready | On a rented A100 80GB PCIe in the pinned container, measure per-epoch and fixed preparation time for each frontier width with telemetry, re-time a ResNet-9 reimplementation against the 59.3 s baseline, and compare a reference configuration's 10-seed accuracy with the dev stack. |
| [T-005](tasks/T-005-single-view-width-by-epochs-frontier-p1.md) — Single-view width by epochs frontier (P1) | exploration | ready | Measure single-view CIFAR-100 accuracy over at least four widths or depths of the airbench-lineage substrate and at least four epoch counts with five seeds per cell, plus a ResNet-9 reference arm, to locate where each width crosses 75.3%. |
| [T-006](tasks/T-006-select-the-base-regime.md) — Select the base regime | decision | proposed | Choose architecture, width, depth, batch size and epoch count by minimum interpolated A100 PCIe time at a 75.3% single-view mean, steelmanning the runner-up regime. |
| [T-007](tasks/T-007-add-on-levers-at-the-selected-base-p2.md) — Add-on levers at the selected base (P2) | exploration | proposed | Compare Muon versus SGD with lookahead, progressive resizing, in-run proxy-loss example selection and label-smoothing level at matched accuracy with at least 10 seeds per arm at the selected base. |
| [T-008](tasks/T-008-adopt-or-reject-add-on-levers.md) — Adopt or reject add-on levers | decision | proposed | Record which levers enter the final recipe and why each rejected lever was rejected. |
| [T-009](tasks/T-009-converge-and-simplify-the-final-submission.md) — Converge and simplify the final submission | delivery | proposed | Remove exploration-only parameters and code paths, make the defaults run the selected recipe, write the submission README, then run one simplify-codebase pass over the integrated team folder. |
| [T-010](tasks/T-010-official-equivalent-40-seed-qualification-on-the-a100-pcie.md) — Official-equivalent 40-seed qualification on the A100 PCIe | assurance | proposed | Run the frozen candidate in the pinned container on an A100 80GB PCIe with a private 40-seed file, cpus 4 and network none; record results, telemetry and the R-001 risk calculation. |
| [T-011](tasks/T-011-fresh-context-compliance-review-and-independence-checks.md) — Fresh-context compliance review and independence checks | assurance | proposed | Review the frozen candidate source against every RULES.md section 3 bullet without implementation narrative, and run repeat-seed, reordered-seed and fresh-process independence checks. |
| [T-012](tasks/T-012-open-the-upstream-pull-request-after-approval.md) — Open the upstream pull request after approval | delivery | proposed | After a recorded team-lead approval, create a clean branch from upstream main containing only the team folder and open the pull request. |

## Issues identified

- No issues recorded.

## Decisions and changes

- No decisions recorded.

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
| R-005 | core | A100 80GB PCIe per-epoch and fixed preparation times are measured in the pinned container for every frontier candidate, and the final recipe's 40-trial mean time is recorded with its standard deviation and GPU telemetry. | pending_evidence |
| R-006 | supporting | Any configuration in the exploration space runs from a declarative sweep file on any free local GPU through the official harness, resumes after interruption without repeating completed runs, and its collated tables are reproducible from raw results. | pending_evidence |
| R-007 | supporting | Single-view accuracy measured on the local development stack agrees with the pinned A100 stack within two standard errors for a reference configuration, or the discrepancy is quantified and applied as a correction. | pending_evidence |
| R-008 | core | The pull request to the upstream repository adds only submissions/<team>/ with its source and README, runs its final recipe from default settings, contains no weights, data or results, and was approved by the team lead before opening. | pending_evidence |

### Worksets

| Sequence | Workset | Decision boundary | Status |
| ---: | --- | --- | --- |
| 1 | Exploration enablement | Can the team run the exploration grid at scale and trust its outputs? | active |
| 2 | Width by epochs frontier | Is the competition capacity-bound or throughput-bound, and which base regime wins? | planned |
| 3 | A100 PCIe calibration | Which time model and which accuracy correction apply to frontier decisions? | planned |
| 4 | Add-on levers at the selected base | Which levers enter the final recipe? | planned |
| 5 | Convergence and assurance | Is the exact candidate qualifying, compliant and ready to submit? | planned |
| 6 | Submission | Has the approved candidate been submitted in the required form? | planned |

### Explorations

- **X-001:** Where does the single-view width by epochs frontier cross 75% on CIFAR-100, and is the minimum-time regime capacity-bound or throughput-bound? (test_ready; test-ready=true)

### Exact frontier

- State: **continue**
- Active: T-001
- Eligible: none
- Unresolved outcomes: O-001, O-002, O-003, O-004
- Unresolved requirements: R-001, R-002, R-003, R-004, R-005, R-006, R-007, R-008
- Pending assessments: none
- Human engagement: none
<!-- writing-tools:generated:end -->
