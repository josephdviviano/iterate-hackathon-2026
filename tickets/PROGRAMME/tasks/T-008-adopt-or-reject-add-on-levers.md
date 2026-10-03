---
schema_version: '2.0'
task_id: T-008
title: Adopt or reject add-on levers
work_type: decision
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
objective: Record which levers enter the final recipe and why each rejected lever was rejected.
consequence: Fixes the recipe content to be converged and assured.
decision_effect: Freezes the recipe content for convergence.
uncertainty: null
dependencies:
- T-007
status: cancelled
acceptance:
- criterion_id: AC-01
  statement: A decision record lists adopted and rejected levers with evidence and reopening conditions
  expected_evidence: decision record
  status: pending
  evidence_ids: []
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence: []
completion_summary: Superseded by T-013 decisions D-003 to D-007, which record adopted and rejected levers
  with evidence and reopening conditions.
blocker_ids: []
---

# T-008: Adopt or reject add-on levers

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Record which levers enter the final recipe and why each rejected lever was rejected.

## Consequence

Fixes the recipe content to be converged and assured.

## Decision effect

Freezes the recipe content for convergence.

## Acceptance

- [ ] AC-01: A decision record lists adopted and rejected levers with evidence and reopening conditions
  Evidence: decision record

## Resolution

Superseded by T-013 decisions D-003 to D-007, which record adopted and rejected levers with evidence and reopening conditions.
