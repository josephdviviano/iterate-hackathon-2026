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
status: ready
acceptance:
- criterion_id: AC-01
  statement: CPU synthetic smoke and repeat-seed reset tests pass through the official harness
  expected_evidence: pytest output and harness summary
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: A real-data GPU run at the airbench94-equivalent configuration completes with recorded single-view
    accuracy and no evaluation recompilation
  expected_evidence: harness summary.json
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-03
  statement: Defaults reproduce a documented reference configuration and unknown parameters fail loudly
  expected_evidence: test output
  status: pending
  evidence_ids: []
  rationale: null
evidence_requirements: []
out_of_scope: []
risks:
- Porting airbench to 100 classes may need retuned head scaling or learning rates before the frontier
  is meaningful.
evidence: []
completion_summary: null
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

- [ ] AC-01: CPU synthetic smoke and repeat-seed reset tests pass through the official harness
  Evidence: pytest output and harness summary
- [ ] AC-02: A real-data GPU run at the airbench94-equivalent configuration completes with recorded single-view accuracy and no evaluation recompilation
  Evidence: harness summary.json
- [ ] AC-03: Defaults reproduce a documented reference configuration and unknown parameters fail loudly
  Evidence: test output

## Risks

- Porting airbench to 100 classes may need retuned head scaling or learning rates before the frontier is meaningful.
