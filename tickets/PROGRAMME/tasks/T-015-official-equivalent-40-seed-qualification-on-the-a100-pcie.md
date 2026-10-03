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
status: completed
acceptance:
- criterion_id: AC-01
  statement: summary.json reports complete with no evaluation timeout and a margin meeting the R-001 risk
    bound
  expected_evidence: summary.json and risk calculation
  status: verified
  evidence_ids:
  - E-001
  - E-002
  rationale: null
- criterion_id: AC-02
  statement: Mean prepare+train time and its sd are recorded with per-trial telemetry
  expected_evidence: trials.jsonl and telemetry table
  status: verified
  evidence_ids:
  - E-001
  - E-002
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence:
- evidence_id: E-001
  kind: file
  label: official-equivalent.txt
  locator: research/evidence/T-015/official-equivalent.txt
  sha256: 56bdfb5d21f03166cf0e929baee179e9b5f33f166e9171fecdf7b07f2af27838
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: m4a-table.csv
  locator: research/evidence/T-015/m4a-table.csv
  sha256: cef3a57d5919063529d967f25976b5e6c258c2bdf83c86f3040b77829ffec735
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: 'Qualifies on A100 80GB PCIe: 75.272% over 40 fresh seeds at 6.027 s per trial, risk
  about 2e-7; telemetry recorded.'
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

- [x] AC-01: summary.json reports complete with no evaluation timeout and a margin meeting the R-001 risk bound
  Evidence: summary.json and risk calculation
- [x] AC-02: Mean prepare+train time and its sd are recorded with per-trial telemetry
  Evidence: trials.jsonl and telemetry table

## Evidence

- E-001: official-equivalent.txt — `research/evidence/T-015/official-equivalent.txt` (verified)
- E-002: m4a-table.csv — `research/evidence/T-015/m4a-table.csv` (verified)

## Resolution

Qualifies on A100 80GB PCIe: 75.272% over 40 fresh seeds at 6.027 s per trial, risk about 2e-7; telemetry recorded.
