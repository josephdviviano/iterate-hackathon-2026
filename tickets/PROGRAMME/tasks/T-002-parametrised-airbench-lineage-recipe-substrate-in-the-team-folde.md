---
schema_version: '2.0'
task_id: T-002
title: Parametrised airbench-lineage recipe substrate in the team folder
work_type: delivery
commitment: required
workset_id: enablement
requirement_ids:
- R-006
scenario_ids:
- S-001
- S-003
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids:
- decision-review
objective: Implement build/prepare/train for an airbench-lineage CIFAR-100 recipe whose architecture,
  width, depth, epochs, batch size, optimiser, schedule and augmentation are selected by declarative parameters,
  with compliance invariants (synthetic-only build, in-place per-trial reset, single-view stateless forward).
consequence: Every frontier and add-on probe runs through this substrate; a compliance or reset defect
  would contaminate all downstream evidence.
decision_effect: Defines the explorable design space for P1 and P2.
uncertainty: null
dependencies:
- T-001
status: completed
acceptance:
- criterion_id: AC-01
  statement: CPU synthetic smoke and repeat-seed reset tests pass through the official harness
  expected_evidence: pytest output and harness summary
  status: verified
  evidence_ids:
  - E-001
  - E-002
  rationale: null
- criterion_id: AC-02
  statement: A real-data GPU run at the airbench94-equivalent configuration completes with recorded single-view
    accuracy and no evaluation recompilation
  expected_evidence: harness summary.json
  status: verified
  evidence_ids:
  - E-001
  - E-002
  rationale: null
- criterion_id: AC-03
  statement: Defaults reproduce a documented reference configuration and unknown parameters fail loudly
  expected_evidence: test output
  status: verified
  evidence_ids:
  - E-001
  - E-002
  rationale: null
evidence_requirements: []
out_of_scope: []
risks:
- Porting airbench to 100 classes may need retuned head scaling or learning rates before the frontier
  is meaningful.
evidence:
- evidence_id: E-001
  kind: file
  label: real-data-default.txt
  locator: research/evidence/T-002/real-data-default.txt
  sha256: 354fde5b5d6f657423ee18fb60d446747e155db6dad699f97e49c15233f81aeb
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: test_recipe.py
  locator: research/tests/test_recipe.py
  sha256: 327449a2bdbc5e977c513e5ad581170f51a6c95da3746025892b9e4b15b762bc
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: Parametrised recipe passes CPU contract tests (reset, invalid params, defaults, harness
  smoke for three variants) on the pinned stack and trains on real data (default 69.5% single-view, eager
  evaluation).
blocker_ids: []
---

# T-002: Parametrised airbench-lineage recipe substrate in the team folder

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Implement build/prepare/train for an airbench-lineage CIFAR-100 recipe whose architecture, width, depth, epochs, batch size, optimiser, schedule and augmentation are selected by declarative parameters, with compliance invariants (synthetic-only build, in-place per-trial reset, single-view stateless forward).

## Consequence

Every frontier and add-on probe runs through this substrate; a compliance or reset defect would contaminate all downstream evidence.

## Decision effect

Defines the explorable design space for P1 and P2.

## Acceptance

- [x] AC-01: CPU synthetic smoke and repeat-seed reset tests pass through the official harness
  Evidence: pytest output and harness summary
- [x] AC-02: A real-data GPU run at the airbench94-equivalent configuration completes with recorded single-view accuracy and no evaluation recompilation
  Evidence: harness summary.json
- [x] AC-03: Defaults reproduce a documented reference configuration and unknown parameters fail loudly
  Evidence: test output

## Risks

- Porting airbench to 100 classes may need retuned head scaling or learning rates before the frontier is meaningful.

## Evidence

- E-001: real-data-default.txt — `research/evidence/T-002/real-data-default.txt` (verified)
- E-002: test_recipe.py — `research/tests/test_recipe.py` (verified)

## Resolution

Parametrised recipe passes CPU contract tests (reset, invalid params, defaults, harness smoke for three variants) on the pinned stack and trains on real data (default 69.5% single-view, eager evaluation).
