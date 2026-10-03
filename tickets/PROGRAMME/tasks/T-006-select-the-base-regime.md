---
schema_version: '2.0'
task_id: T-006
title: Select the base regime
work_type: decision
commitment: required
workset_id: frontier
requirement_ids:
- R-003
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids: []
objective: Choose architecture, width, depth, batch size and epoch count by minimum interpolated A100
  PCIe time at a 75.3% single-view mean, steelmanning the runner-up regime.
consequence: Every add-on comparison and the final recipe inherit this base.
decision_effect: Fixes the base for the add-ons workset and states reopening conditions.
uncertainty: null
dependencies:
- T-004
- T-005
status: proposed
acceptance:
- criterion_id: AC-01
  statement: A decision record selects the regime from the frontier table and A100 timings, with rejected
    alternatives and reopening conditions
  expected_evidence: decision record
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

# T-006: Select the base regime

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Choose architecture, width, depth, batch size and epoch count by minimum interpolated A100 PCIe time at a 75.3% single-view mean, steelmanning the runner-up regime.

## Consequence

Every add-on comparison and the final recipe inherit this base.

## Decision effect

Fixes the base for the add-ons workset and states reopening conditions.

## Acceptance

- [ ] AC-01: A decision record selects the regime from the frontier table and A100 timings, with rejected alternatives and reopening conditions
  Evidence: decision record
