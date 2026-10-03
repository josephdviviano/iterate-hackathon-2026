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
status: ready
acceptance:
- criterion_id: AC-01
  statement: Exploration portfolio X-001 with hypotheses H1 to H6 validates
  expected_evidence: programme explore-check output
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: The frontier table with mean, sd and n per cell is reproducible from raw harness results
  expected_evidence: collation command and table
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-03
  statement: A finding interprets the frontier, including any cell within two standard errors of 75.3%
  expected_evidence: finding record
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

# T-005: Single-view width by epochs frontier (P1)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Measure single-view CIFAR-100 accuracy over at least four widths or depths of the airbench-lineage substrate and at least four epoch counts with five seeds per cell, plus a ResNet-9 reference arm, to locate where each width crosses 75.3%.

## Consequence

Optimiser and kernel investment before this probe risks optimising the wrong regime.

## Decision effect

Decides whether the competition is capacity-bound or throughput-bound and supplies E*(w) for the regime decision.

## Acceptance

- [ ] AC-01: Exploration portfolio X-001 with hypotheses H1 to H6 validates
  Evidence: programme explore-check output
- [ ] AC-02: The frontier table with mean, sd and n per cell is reproducible from raw harness results
  Evidence: collation command and table
- [ ] AC-03: A finding interprets the frontier, including any cell within two standard errors of 75.3%
  Evidence: finding record
