---
schema_version: '2.0'
task_id: T-007
title: Add-on levers at the selected base (P2)
work_type: exploration
commitment: required
workset_id: addons
requirement_ids:
- R-004
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids: []
objective: Compare Muon versus SGD with lookahead, progressive resizing, in-run proxy-loss example selection
  and label-smoothing level at matched accuracy with at least 10 seeds per arm at the selected base.
consequence: Unmeasured levers either leave time on the table or add complexity that does not pay.
decision_effect: Determines which levers enter the final recipe.
uncertainty: Whether levers measured on TTA-scored CIFAR-10 records reduce time at matched single-view
  CIFAR-100 accuracy.
dependencies:
- T-006
status: cancelled
acceptance:
- criterion_id: AC-01
  statement: Each lever has a matched-accuracy time comparison with confidence intervals
  expected_evidence: collated comparison table
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: A finding interprets each lever, including null results
  expected_evidence: finding records
  status: pending
  evidence_ids: []
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence: []
completion_summary: 'Superseded by T-013: probes S4-S15 tested Muon, resizing, selection, batch size,
  regularisation, compile and further levers at the climbed base (F-007 to F-019); its premise of a separate
  add-on pass after an A100 regime decision no longer holds because the climb ran on the local proxy under
  D-001.'
blocker_ids: []
---

# T-007: Add-on levers at the selected base (P2)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Compare Muon versus SGD with lookahead, progressive resizing, in-run proxy-loss example selection and label-smoothing level at matched accuracy with at least 10 seeds per arm at the selected base.

## Consequence

Unmeasured levers either leave time on the table or add complexity that does not pay.

## Decision effect

Determines which levers enter the final recipe.

## Acceptance

- [ ] AC-01: Each lever has a matched-accuracy time comparison with confidence intervals
  Evidence: collated comparison table
- [ ] AC-02: A finding interprets each lever, including null results
  Evidence: finding records

## Resolution

Superseded by T-013: probes S4-S15 tested Muon, resizing, selection, batch size, regularisation, compile and further levers at the climbed base (F-007 to F-019); its premise of a separate add-on pass after an A100 regime decision no longer holds because the climb ran on the local proxy under D-001.
