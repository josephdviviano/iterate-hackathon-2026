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
status: completed
acceptance:
- criterion_id: AC-01
  statement: Per-epoch and fixed-cost A100 PCIe timings with SM clock, power and temperature telemetry
    exist for each frontier width
  expected_evidence: harness results and telemetry table
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
- criterion_id: AC-02
  statement: The cross-stack 10-seed comparison is reported with combined standard errors
  expected_evidence: comparison table and finding
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
evidence_requirements: []
out_of_scope: []
risks:
- Requires team-lead-approved A100 rental; record an external blocker with a retry condition until access
  exists.
evidence:
- evidence_id: E-001
  kind: file
  label: X-004-what-are-a100-80gb-pcie-timings-for-the-frontier-candidates-and-.json
  locator: tickets/PROGRAMME/explorations/X-004-what-are-a100-80gb-pcie-timings-for-the-frontier-candidates-and-.json
  sha256: 2fc4a97a480224ad76211f3d639aca058ece114b7239f6024800a91ee887e599
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: m1-a100-details.txt
  locator: research/evidence/T-004/m1-a100-details.txt
  sha256: f149a85da8d3680660102c8febeaf42c41afa3f8a18fdc5ac5c4c485733a02c0
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: m2-a100-cudagraphs-table.csv
  locator: research/evidence/T-004/m2-a100-cudagraphs-table.csv
  sha256: 1b4fcdf600a229d6ab6ce9c76c15e4bc5fe3bdff0c3beca3fbeffc52b80955aa
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: cross-stack-pairs.txt
  locator: research/evidence/T-004/cross-stack-pairs.txt
  sha256: 8959a6121066aa47c1a27dbe9a2e9902889a6644747669ee94e669e100a27ee0
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: 'A100 80GB PCIe: 5.76 s (512), 6.17-6.26 s (640, default compile), 6.12 s with max-autotune;
  cross-stack accuracy agrees within 0.1 pp over 120 paired seeds.'
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

- [x] AC-01: Per-epoch and fixed-cost A100 PCIe timings with SM clock, power and temperature telemetry exist for each frontier width
  Evidence: harness results and telemetry table
- [x] AC-02: The cross-stack 10-seed comparison is reported with combined standard errors
  Evidence: comparison table and finding

## Risks

- Requires team-lead-approved A100 rental; record an external blocker with a retry condition until access exists.

## Evidence

- E-001: X-004-what-are-a100-80gb-pcie-timings-for-the-frontier-candidates-and-.json — `tickets/PROGRAMME/explorations/X-004-what-are-a100-80gb-pcie-timings-for-the-frontier-candidates-and-.json` (verified)
- E-002: m1-a100-details.txt — `research/evidence/T-004/m1-a100-details.txt` (verified)
- E-003: m2-a100-cudagraphs-table.csv — `research/evidence/T-004/m2-a100-cudagraphs-table.csv` (verified)
- E-004: cross-stack-pairs.txt — `research/evidence/T-004/cross-stack-pairs.txt` (verified)

## Resolution

A100 80GB PCIe: 5.76 s (512), 6.17-6.26 s (640, default compile), 6.12 s with max-autotune; cross-stack accuracy agrees within 0.1 pp over 120 paired seeds.
