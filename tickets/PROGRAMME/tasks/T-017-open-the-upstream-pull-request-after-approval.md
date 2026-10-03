---
schema_version: '2.0'
task_id: T-017
title: Open the upstream pull request after approval
work_type: delivery
commitment: required
workset_id: release
requirement_ids:
- R-008
scenario_ids:
- S-006
assumption_ids: []
affected_boundaries:
- upstream-pull-request
review_path_ids:
- release-review
objective: After a recorded team-lead approval, create a clean branch from upstream main containing only
  the team folder and open the pull request.
consequence: The entry is not judged unless the pull request exists in the required form.
decision_effect: Releases the entry to the organisers.
uncertainty: null
dependencies:
- T-015
- T-016
status: proposed
acceptance:
- criterion_id: AC-01
  statement: A human decision approving submission is recorded
  expected_evidence: human-decision record
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: The pull request diff contains only submissions/<team>/
  expected_evidence: pull request file list
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

# T-017: Open the upstream pull request after approval

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

After a recorded team-lead approval, create a clean branch from upstream main containing only the team folder and open the pull request.

## Consequence

The entry is not judged unless the pull request exists in the required form.

## Decision effect

Releases the entry to the organisers.

## Acceptance

- [ ] AC-01: A human decision approving submission is recorded
  Evidence: human-decision record
- [ ] AC-02: The pull request diff contains only submissions/<team>/
  Evidence: pull request file list
