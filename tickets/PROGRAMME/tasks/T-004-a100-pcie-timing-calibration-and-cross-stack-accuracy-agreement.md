---
schema_version: '2.0'
task_id: T-004
title: A100 PCIe timing calibration and cross-stack accuracy agreement
work_type: exploration
commitment: required
workset_id: calibration
requirement_ids:
- R-005
- R-007
scenario_ids:
- S-001
- S-002
assumption_ids: []
affected_boundaries:
- a100-container
review_path_ids: []
objective: On a rented A100 80GB PCIe in the pinned container, measure per-epoch and fixed preparation
  time for each frontier width with telemetry, re-time a ResNet-9 reimplementation against the 59.3 s
  baseline, and compare a reference configuration's 10-seed accuracy with the dev stack.
consequence: Frontier cells cannot be converted into times, and local accuracy cannot be trusted near
  the threshold, without this calibration.
decision_effect: Supplies the time model and any accuracy correction for the regime decision.
uncertainty: Per-epoch A100 PCIe time per candidate width under 300 W power capping, and whether torch
  2.7.1 Blackwell accuracy transfers to torch 2.4.0 A100 accuracy.
dependencies:
- T-002
status: ready
acceptance:
- criterion_id: AC-01
  statement: Per-epoch and fixed-cost A100 PCIe timings with SM clock, power and temperature telemetry
    exist for each frontier width
  expected_evidence: harness results and telemetry table
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: The cross-stack 10-seed comparison is reported with combined standard errors
  expected_evidence: comparison table and finding
  status: pending
  evidence_ids: []
  rationale: null
evidence_requirements: []
out_of_scope: []
risks:
- Requires team-lead-approved A100 rental; record an external blocker with a retry condition until access
  exists.
evidence: []
completion_summary: null
blocker_ids: []
---

# T-004: A100 PCIe timing calibration and cross-stack accuracy agreement

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

On a rented A100 80GB PCIe in the pinned container, measure per-epoch and fixed preparation time for each frontier width with telemetry, re-time a ResNet-9 reimplementation against the 59.3 s baseline, and compare a reference configuration's 10-seed accuracy with the dev stack.

## Consequence

Frontier cells cannot be converted into times, and local accuracy cannot be trusted near the threshold, without this calibration.

## Decision effect

Supplies the time model and any accuracy correction for the regime decision.

## Acceptance

- [ ] AC-01: Per-epoch and fixed-cost A100 PCIe timings with SM clock, power and temperature telemetry exist for each frontier width
  Evidence: harness results and telemetry table
- [ ] AC-02: The cross-stack 10-seed comparison is reported with combined standard errors
  Evidence: comparison table and finding

## Risks

- Requires team-lead-approved A100 rental; record an external blocker with a retry condition until access exists.
