---
schema_version: '2.0'
task_id: T-009
title: Converge and simplify the final submission
work_type: delivery
commitment: required
workset_id: assurance
requirement_ids:
- R-002
scenario_ids:
- S-005
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids:
- compliance-review
objective: Remove exploration-only parameters and code paths, make the defaults run the selected recipe,
  write the submission README, then run one simplify-codebase pass over the integrated team folder.
consequence: Exploration flags left in the submission enlarge the review surface and can change behaviour.
decision_effect: Produces the exact candidate for qualification and compliance assurance.
uncertainty: null
dependencies:
- T-008
status: proposed
acceptance:
- criterion_id: AC-01
  statement: Default settings reproduce the selected recipe's dev-stack accuracy within noise over 10
    seeds
  expected_evidence: harness summary
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: The team folder contains only source and README, with no exploration-only parameters
  expected_evidence: diff inspection
  status: pending
  evidence_ids: []
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence: []
completion_summary: null
blocker_ids: []
---

# T-009: Converge and simplify the final submission

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Remove exploration-only parameters and code paths, make the defaults run the selected recipe, write the submission README, then run one simplify-codebase pass over the integrated team folder.

## Consequence

Exploration flags left in the submission enlarge the review surface and can change behaviour.

## Decision effect

Produces the exact candidate for qualification and compliance assurance.

## Acceptance

- [ ] AC-01: Default settings reproduce the selected recipe's dev-stack accuracy within noise over 10 seeds
  Evidence: harness summary
- [ ] AC-02: The team folder contains only source and README, with no exploration-only parameters
  Evidence: diff inspection
