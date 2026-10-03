---
schema_version: '2.0'
task_id: T-015
title: Official-equivalent 40-seed qualification on the A100 PCIe
work_type: assurance
commitment: required
workset_id: assurance
requirement_ids:
- R-001
- R-005
- R-007
scenario_ids:
- S-002
- S-001
assumption_ids: []
affected_boundaries:
- a100-container
review_path_ids:
- qualification-review
objective: Run the frozen converged candidate in the pinned container on an A100 80GB PCIe with a private
  40-seed file, cpus 4 and network none; record results, telemetry, the R-001 risk calculation and the
  dev-stack versus A100 accuracy comparison; fall back to 8.5 or 8.75 epochs (D-007) if the mean is below
  75.2%.
consequence: Without it neither qualification nor the score is established.
decision_effect: Decides whether the candidate can be submitted and at which budget.
uncertainty: null
dependencies:
- T-014
- T-004
status: proposed
acceptance:
- criterion_id: AC-01
  statement: summary.json reports complete with no evaluation timeout and a margin meeting the R-001 risk
    bound
  expected_evidence: summary.json and risk calculation
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: Mean prepare+train time and its sd are recorded with per-trial telemetry
  expected_evidence: trials.jsonl and telemetry table
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

# T-015: Official-equivalent 40-seed qualification on the A100 PCIe

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Run the frozen converged candidate in the pinned container on an A100 80GB PCIe with a private 40-seed file, cpus 4 and network none; record results, telemetry, the R-001 risk calculation and the dev-stack versus A100 accuracy comparison; fall back to 8.5 or 8.75 epochs (D-007) if the mean is below 75.2%.

## Consequence

Without it neither qualification nor the score is established.

## Decision effect

Decides whether the candidate can be submitted and at which budget.

## Acceptance

- [ ] AC-01: summary.json reports complete with no evaluation timeout and a margin meeting the R-001 risk bound
  Evidence: summary.json and risk calculation
- [ ] AC-02: Mean prepare+train time and its sd are recorded with per-trial telemetry
  Evidence: trials.jsonl and telemetry table
