---
schema_version: '2.0'
task_id: T-013
title: Local-proxy hill climb to convergence (X-002)
work_type: exploration
commitment: required
workset_id: frontier
requirement_ids:
- R-003
- R-004
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids: []
objective: From the P1 frontier, climb one lever at a time (width and stage allocation, epochs, optimiser,
  batch size, resolution schedule, regularisation, learning rates) on GPU 1, keeping a change only when
  the interpolated proxy time to a 75.3% single-view mean falls beyond seed noise; stop when no accessible
  lever improves it.
consequence: Without a systematic climb the submitted recipe stays at the first frontier point and leaves
  time on the table.
decision_effect: Produces the provisional best recipe and the evidence for D-series selections, all reopening
  on A100 calibration.
uncertainty: Which levers lower local proxy time at matched single-view accuracy once the base regime
  is wide and short.
dependencies:
- T-005
status: in_progress
acceptance:
- criterion_id: AC-01
  statement: Exploration portfolio X-002 validates and ends saturated or closed
  expected_evidence: programme explore-check output
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: Each climb step has a collated sweep table and a finding or decision
  expected_evidence: sweep tables and programme records
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-03
  statement: The provisional best recipe reaches a single-view mean of at least 75.3% over at least 20
    seeds
  expected_evidence: collated confirmation table
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

# T-013: Local-proxy hill climb to convergence (X-002)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

From the P1 frontier, climb one lever at a time (width and stage allocation, epochs, optimiser, batch size, resolution schedule, regularisation, learning rates) on GPU 1, keeping a change only when the interpolated proxy time to a 75.3% single-view mean falls beyond seed noise; stop when no accessible lever improves it.

## Consequence

Without a systematic climb the submitted recipe stays at the first frontier point and leaves time on the table.

## Decision effect

Produces the provisional best recipe and the evidence for D-series selections, all reopening on A100 calibration.

## Acceptance

- [ ] AC-01: Exploration portfolio X-002 validates and ends saturated or closed
  Evidence: programme explore-check output
- [ ] AC-02: Each climb step has a collated sweep table and a finding or decision
  Evidence: sweep tables and programme records
- [ ] AC-03: The provisional best recipe reaches a single-view mean of at least 75.3% over at least 20 seeds
  Evidence: collated confirmation table
