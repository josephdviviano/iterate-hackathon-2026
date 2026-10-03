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
status: in_progress
acceptance:
- criterion_id: AC-01
  statement: Every strategy in the coverage matrix has a tested, rejected-with-evidence or not-applicable
    disposition
  expected_evidence: research/strategy-coverage.md
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: Any lever beating control by more than two standard errors is confirmed with fresh seeds
    or rejected
  expected_evidence: collated tables and findings
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

# T-018: Coverage pass: every lineage, competitor and novel strategy (X-003)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Test every untested strategy from the CIFAR-10 record lineage (airbench, airbench94_muon, airbench96_faster, hiverge, Fable/Fulcrum, hlb-CIFAR10, Page), the competitor's programme and novel alternatives against the converged recipe; climb any that beats control beyond noise and record a disposition for each.

## Consequence

An untested lineage strategy could beat the converged recipe; untested items leave the convergence claim incomplete.

## Decision effect

Either reopens the climb with a new lever or confirms saturation against the full known strategy space.

## Acceptance

- [ ] AC-01: Every strategy in the coverage matrix has a tested, rejected-with-evidence or not-applicable disposition
  Evidence: research/strategy-coverage.md
- [ ] AC-02: Any lever beating control by more than two standard errors is confirmed with fresh seeds or rejected
  Evidence: collated tables and findings
