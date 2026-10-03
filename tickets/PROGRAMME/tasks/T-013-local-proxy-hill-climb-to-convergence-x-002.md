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
status: completed
acceptance:
- criterion_id: AC-01
  statement: Exploration portfolio X-002 validates and ends saturated or closed
  expected_evidence: programme explore-check output
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  - E-005
  - E-006
  rationale: null
- criterion_id: AC-02
  statement: Each climb step has a collated sweep table and a finding or decision
  expected_evidence: sweep tables and programme records
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  - E-005
  - E-006
  rationale: null
- criterion_id: AC-03
  statement: The provisional best recipe reaches a single-view mean of at least 75.3% over at least 20
    seeds
  expected_evidence: collated confirmation table
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  - E-005
  - E-006
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence:
- evidence_id: E-001
  kind: file
  label: X-002-which-levers-lower-local-proxy-time-to-a-75-3-single-view-mean-f.json
  locator: tickets/PROGRAMME/explorations/X-002-which-levers-lower-local-proxy-time-to-a-75-3-single-view-mean-f.json
  sha256: 7e5544d50ea15a7c763a2049451765d273d096566447099c07b3d724e80df3b3
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: s13-base640-epochs-table.csv
  locator: research/evidence/T-013/s13-base640-epochs-table.csv
  sha256: de798712fc194729937940676cd183c671aaabb71a98964fb8d5b18e9aa6aa7a
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: s15-logit-scale-table.csv
  locator: research/evidence/T-013/s15-logit-scale-table.csv
  sha256: 56d41cab13d0921d426efe846947a4a6954d0a68041a93c40d748cd6af4681e5
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: s19-switch-point-table.csv
  locator: research/evidence/T-013/s19-switch-point-table.csv
  sha256: 8b6cc8599f7345c821f599814ee0e2c1d29ce1c86f78b466c505e18f53ac8eff
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-005
  kind: file
  label: s18-shorter40-table.csv
  locator: research/evidence/T-013/s18-shorter40-table.csv
  sha256: 6ccf8ad97026fcfcc842a2ef3017173536b24d9cbe640cf6fe89fcbdd777088d
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-006
  kind: file
  label: s16-confirm40-table.csv
  locator: research/evidence/T-013/s16-confirm40-table.csv
  sha256: 6c3842a524774f597aec66b1ed7a9420ff86fb80fdd8d58057605c6a26ea9bc7
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: 'Converged after S3-S19: 128/384/640 depth-3, translate 2, logit scale 1/6, 20/32
  px resizing, compiled; 40-seed fresh means 75.33% at 8.5 and 75.43% at 8.75 epochs (AC-03), budget 8.25
  epochs (75.26%) selected under F-005 (D-007); local compiled proxy 5.38 s vs 9.3 s for the P1 best cell.'
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

- [x] AC-01: Exploration portfolio X-002 validates and ends saturated or closed
  Evidence: programme explore-check output
- [x] AC-02: Each climb step has a collated sweep table and a finding or decision
  Evidence: sweep tables and programme records
- [x] AC-03: The provisional best recipe reaches a single-view mean of at least 75.3% over at least 20 seeds
  Evidence: collated confirmation table

## Evidence

- E-001: X-002-which-levers-lower-local-proxy-time-to-a-75-3-single-view-mean-f.json — `tickets/PROGRAMME/explorations/X-002-which-levers-lower-local-proxy-time-to-a-75-3-single-view-mean-f.json` (verified)
- E-002: s13-base640-epochs-table.csv — `research/evidence/T-013/s13-base640-epochs-table.csv` (verified)
- E-003: s15-logit-scale-table.csv — `research/evidence/T-013/s15-logit-scale-table.csv` (verified)
- E-004: s19-switch-point-table.csv — `research/evidence/T-013/s19-switch-point-table.csv` (verified)
- E-005: s18-shorter40-table.csv — `research/evidence/T-013/s18-shorter40-table.csv` (verified)
- E-006: s16-confirm40-table.csv — `research/evidence/T-013/s16-confirm40-table.csv` (verified)

## Resolution

Converged after S3-S19: 128/384/640 depth-3, translate 2, logit scale 1/6, 20/32 px resizing, compiled; 40-seed fresh means 75.33% at 8.5 and 75.43% at 8.75 epochs (AC-03), budget 8.25 epochs (75.26%) selected under F-005 (D-007); local compiled proxy 5.38 s vs 9.3 s for the P1 best cell.
