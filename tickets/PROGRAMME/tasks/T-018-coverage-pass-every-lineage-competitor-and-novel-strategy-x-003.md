---
schema_version: '2.0'
task_id: T-018
title: 'Coverage pass: every lineage, competitor and novel strategy (X-003)'
work_type: exploration
commitment: required
workset_id: frontier
requirement_ids:
- R-004
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids: []
objective: Test every untested strategy from the CIFAR-10 record lineage (airbench, airbench94_muon, airbench96_faster,
  hiverge, Fable/Fulcrum, hlb-CIFAR10, Page), the competitor's programme and novel alternatives against
  the converged recipe; climb any that beats control beyond noise and record a disposition for each.
consequence: An untested lineage strategy could beat the converged recipe; untested items leave the convergence
  claim incomplete.
decision_effect: Either reopens the climb with a new lever or confirms saturation against the full known
  strategy space.
uncertainty: Whether strategies tuned for TTA-scored CIFAR-10 records (hiverge Muon stack, colour jitter,
  activations, pooling, execution tricks) or novel alternatives improve single-view CIFAR-100 time to
  target.
dependencies:
- T-013
status: completed
acceptance:
- criterion_id: AC-01
  statement: Every strategy in the coverage matrix has a tested, rejected-with-evidence or not-applicable
    disposition
  expected_evidence: research/strategy-coverage.md
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  - E-005
  rationale: null
- criterion_id: AC-02
  statement: Any lever beating control by more than two standard errors is confirmed with fresh seeds
    or rejected
  expected_evidence: collated tables and findings
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  - E-005
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence:
- evidence_id: E-001
  kind: file
  label: X-003-do-representation-or-learning-approaches-outside-the-airbench-co.json
  locator: tickets/PROGRAMME/explorations/X-003-do-representation-or-learning-approaches-outside-the-airbench-co.json
  sha256: 25c9c0f991111eb8e61b79832ca1a93e1d8d720bd74e88f2e2e95797b2996077
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: strategy-coverage-final.md
  locator: research/evidence/T-018/strategy-coverage-final.md
  sha256: 86a2783d6a33d484c1bf6eed4c604a45d4565ae74e66f6549bf4c057e0f65fc0
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: s27-momentum-table.csv
  locator: research/evidence/T-018/s27-momentum-table.csv
  sha256: ac876d918100535ffe0ca4e6613575fa1a462281594db3b6210dfbe2f11bc99d
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: s26a-timing-table.csv
  locator: research/evidence/T-018/s26a-timing-table.csv
  sha256: 4678f2c5013d297ebb192632c83cc389eda7bfd886a20198027f41b68b802c54
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-005
  kind: file
  label: s26b-timing-table.csv
  locator: research/evidence/T-018/s26b-timing-table.csv
  sha256: dd461a6274167fe7d6c99b35e2b31019c5972d6166b8bc7f423e9d2d340ce93b
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: Every lineage, competitor and representation strategy has a disposition; only flatten-max
  pooling and max-autotune CUDA graphs survived (about 3% faster at equal accuracy) and are now defaults;
  the momentum candidate failed fresh-seed confirmation.
blocker_ids: []
---

# T-018: Coverage pass: every lineage, competitor and novel strategy (X-003)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Test every untested strategy from the CIFAR-10 record lineage (airbench, airbench94_muon, airbench96_faster, hiverge, Fable/Fulcrum, hlb-CIFAR10, Page), the competitor's programme and novel alternatives against the converged recipe; climb any that beats control beyond noise and record a disposition for each.

## Consequence

An untested lineage strategy could beat the converged recipe; untested items leave the convergence claim incomplete.

## Decision effect

Either reopens the climb with a new lever or confirms saturation against the full known strategy space.

## Acceptance

- [x] AC-01: Every strategy in the coverage matrix has a tested, rejected-with-evidence or not-applicable disposition
  Evidence: research/strategy-coverage.md
- [x] AC-02: Any lever beating control by more than two standard errors is confirmed with fresh seeds or rejected
  Evidence: collated tables and findings

## Evidence

- E-001: X-003-do-representation-or-learning-approaches-outside-the-airbench-co.json — `tickets/PROGRAMME/explorations/X-003-do-representation-or-learning-approaches-outside-the-airbench-co.json` (verified)
- E-002: strategy-coverage-final.md — `research/evidence/T-018/strategy-coverage-final.md` (verified)
- E-003: s27-momentum-table.csv — `research/evidence/T-018/s27-momentum-table.csv` (verified)
- E-004: s26a-timing-table.csv — `research/evidence/T-018/s26a-timing-table.csv` (verified)
- E-005: s26b-timing-table.csv — `research/evidence/T-018/s26b-timing-table.csv` (verified)

## Resolution

Every lineage, competitor and representation strategy has a disposition; only flatten-max pooling and max-autotune CUDA graphs survived (about 3% faster at equal accuracy) and are now defaults; the momentum candidate failed fresh-seed confirmation.
