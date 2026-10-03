---
schema_version: '2.0'
task_id: T-016
title: Fresh-context compliance review and independence checks
work_type: assurance
commitment: required
workset_id: assurance
requirement_ids:
- R-002
scenario_ids:
- S-005
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids:
- compliance-review
objective: Review the frozen converged candidate against every RULES.md section 3 bullet without implementation
  narrative, and run repeat-seed, reordered-seed and fresh-process independence checks.
consequence: A rule violation disqualifies the entry regardless of its time.
decision_effect: Decides whether the candidate is compliant for submission.
uncertainty: null
dependencies:
- T-014
status: ready
acceptance:
- criterion_id: AC-01
  statement: Every RULES.md section 3 bullet has a verdict with source locations
  expected_evidence: review record
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: Independence checks reproduce per-seed accuracy within nondeterminism
  expected_evidence: harness results
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

# T-016: Fresh-context compliance review and independence checks

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Review the frozen converged candidate against every RULES.md section 3 bullet without implementation narrative, and run repeat-seed, reordered-seed and fresh-process independence checks.

## Consequence

A rule violation disqualifies the entry regardless of its time.

## Decision effect

Decides whether the candidate is compliant for submission.

## Acceptance

- [ ] AC-01: Every RULES.md section 3 bullet has a verdict with source locations
  Evidence: review record
- [ ] AC-02: Independence checks reproduce per-seed accuracy within nondeterminism
  Evidence: harness results
