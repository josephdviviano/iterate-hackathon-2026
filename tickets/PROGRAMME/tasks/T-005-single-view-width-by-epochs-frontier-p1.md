---
schema_version: '2.0'
task_id: T-005
title: Single-view width by epochs frontier (P1)
work_type: exploration
commitment: required
workset_id: frontier
requirement_ids:
- R-003
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- exploration-tooling
review_path_ids: []
objective: Measure single-view CIFAR-100 accuracy over at least four widths or depths of the airbench-lineage
  substrate and at least four epoch counts with five seeds per cell, plus a ResNet-9 reference arm, to
  locate where each width crosses 75.3%.
consequence: Optimiser and kernel investment before this probe risks optimising the wrong regime.
decision_effect: Decides whether the competition is capacity-bound or throughput-bound and supplies E*(w)
  for the regime decision.
uncertainty: Where the width by epochs frontier crosses 75% under single-view evaluation on CIFAR-100;
  no published source measures it.
dependencies:
- T-002
- T-003
status: completed
acceptance:
- criterion_id: AC-01
  statement: Exploration portfolio X-001 with hypotheses H1 to H6 validates
  expected_evidence: programme explore-check output
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
- criterion_id: AC-02
  statement: The frontier table with mean, sd and n per cell is reproducible from raw harness results
  expected_evidence: collation command and table
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
- criterion_id: AC-03
  statement: A finding interprets the frontier, including any cell within two standard errors of 75.3%
  expected_evidence: finding record
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence:
- evidence_id: E-001
  kind: file
  label: X-001-where-does-the-single-view-width-by-epochs-frontier-cross-75-on-.json
  locator: tickets/PROGRAMME/explorations/X-001-where-does-the-single-view-width-by-epochs-frontier-cross-75-on-.json
  sha256: 4adf4be354596dd56533a90383494f303c60a6e5ded6796522af277e0dfec837
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: p1-frontier-table.csv
  locator: research/evidence/T-005/p1-frontier-table.csv
  sha256: 536f0ae28919c5c3790f0247b98e761f4eb0efaba979f062b16cf66427daffd8
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: p1-frontier.toml
  locator: research/evidence/T-005/p1-frontier.toml
  sha256: e36d0b62fb284000d1cc9a414a1544e2540a4eefb0df1186fc1d7e8d743c68df
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: F-003-p1-5-seeds-per-cell-single-view-dev-stack-the-airbench94-shape-p.yaml
  locator: tickets/PROGRAMME/findings/F-003-p1-5-seeds-per-cell-single-view-dev-stack-the-airbench94-shape-p.yaml
  sha256: 88286df173999ea4970f526ee4c09a778a2f0615a5f5453b2aafdd3b2d5f9eba
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: Frontier measured over 4 shapes x 4 epoch counts x 5 seeds plus a ResNet-9 arm; airbench96
  shape reaches 75.37% at 10 epochs, 2x crosses at 13.7 epochs; capacity-bound (F-003).
blocker_ids: []
---

# T-005: Single-view width by epochs frontier (P1)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Measure single-view CIFAR-100 accuracy over at least four widths or depths of the airbench-lineage substrate and at least four epoch counts with five seeds per cell, plus a ResNet-9 reference arm, to locate where each width crosses 75.3%.

## Consequence

Optimiser and kernel investment before this probe risks optimising the wrong regime.

## Decision effect

Decides whether the competition is capacity-bound or throughput-bound and supplies E*(w) for the regime decision.

## Acceptance

- [x] AC-01: Exploration portfolio X-001 with hypotheses H1 to H6 validates
  Evidence: programme explore-check output
- [x] AC-02: The frontier table with mean, sd and n per cell is reproducible from raw harness results
  Evidence: collation command and table
- [x] AC-03: A finding interprets the frontier, including any cell within two standard errors of 75.3%
  Evidence: finding record

## Evidence

- E-001: X-001-where-does-the-single-view-width-by-epochs-frontier-cross-75-on-.json — `tickets/PROGRAMME/explorations/X-001-where-does-the-single-view-width-by-epochs-frontier-cross-75-on-.json` (verified)
- E-002: p1-frontier-table.csv — `research/evidence/T-005/p1-frontier-table.csv` (verified)
- E-003: p1-frontier.toml — `research/evidence/T-005/p1-frontier.toml` (verified)
- E-004: F-003-p1-5-seeds-per-cell-single-view-dev-stack-the-airbench94-shape-p.yaml — `tickets/PROGRAMME/findings/F-003-p1-5-seeds-per-cell-single-view-dev-stack-the-airbench94-shape-p.yaml` (verified)

## Resolution

Frontier measured over 4 shapes x 4 epoch counts x 5 seeds plus a ResNet-9 arm; airbench96 shape reaches 75.37% at 10 epochs, 2x crosses at 13.7 epochs; capacity-bound (F-003).
