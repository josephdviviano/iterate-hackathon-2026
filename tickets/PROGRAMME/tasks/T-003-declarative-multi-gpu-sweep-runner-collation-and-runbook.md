---
schema_version: '2.0'
task_id: T-003
title: Declarative multi-GPU sweep runner, collation and runbook
work_type: delivery
commitment: required
workset_id: enablement
requirement_ids:
- R-006
scenario_ids:
- S-001
- S-003
- S-004
- S-007
assumption_ids: []
affected_boundaries:
- exploration-tooling
review_path_ids:
- decision-review
objective: Run sweep files of configurations and seeds through the official harness on free local GPU
  slots, skip completed configurations on resume, record failures, and collate mean, sd and n per configuration
  into reproducible tables; document setup, sweeps, resume and handoff in an executable runbook.
consequence: Scaled exploration across GPUs and agents is impossible or unreliable without idempotent
  dispatch and reproducible collation.
decision_effect: Determines exploration throughput and the trustworthiness of every frontier table.
uncertainty: null
dependencies:
- T-001
status: ready
acceptance:
- criterion_id: AC-01
  statement: A multi-configuration sweep runs across both local GPUs through the harness
  expected_evidence: sweep log and collated table
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: An interrupted sweep resumes without rerunning completed configurations and collation matches
    the uninterrupted table
  expected_evidence: resume log and table diff
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-03
  statement: A failing configuration is reported as failed in collation rather than dropped
  expected_evidence: collated table row
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-04
  statement: The runbook commands execute as written
  expected_evidence: runbook walkthrough output
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

# T-003: Declarative multi-GPU sweep runner, collation and runbook

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Run sweep files of configurations and seeds through the official harness on free local GPU slots, skip completed configurations on resume, record failures, and collate mean, sd and n per configuration into reproducible tables; document setup, sweeps, resume and handoff in an executable runbook.

## Consequence

Scaled exploration across GPUs and agents is impossible or unreliable without idempotent dispatch and reproducible collation.

## Decision effect

Determines exploration throughput and the trustworthiness of every frontier table.

## Acceptance

- [ ] AC-01: A multi-configuration sweep runs across both local GPUs through the harness
  Evidence: sweep log and collated table
- [ ] AC-02: An interrupted sweep resumes without rerunning completed configurations and collation matches the uninterrupted table
  Evidence: resume log and table diff
- [ ] AC-03: A failing configuration is reported as failed in collation rather than dropped
  Evidence: collated table row
- [ ] AC-04: The runbook commands execute as written
  Evidence: runbook walkthrough output
