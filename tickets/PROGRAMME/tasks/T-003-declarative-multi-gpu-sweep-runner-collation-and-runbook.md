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
status: completed
acceptance:
- criterion_id: AC-01
  statement: A multi-configuration sweep runs across both local GPUs through the harness
  expected_evidence: sweep log and collated table
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
  statement: An interrupted sweep resumes without rerunning completed configurations and collation matches
    the uninterrupted table
  expected_evidence: resume log and table diff
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
  statement: A failing configuration is reported as failed in collation rather than dropped
  expected_evidence: collated table row
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  - E-005
  - E-006
  rationale: null
- criterion_id: AC-04
  statement: The runbook commands execute as written
  expected_evidence: runbook walkthrough output
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
  label: smoke-gpu-sweep.log
  locator: research/evidence/T-003/smoke-gpu-sweep.log
  sha256: 724355546a846f2a5ad013cb02480d11fc06e5ff0adef024b90bb4be4a64488e
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: smoke-gpu-table.csv
  locator: research/evidence/T-003/smoke-gpu-table.csv
  sha256: 86cc166242ef9244f41430e095f3f8a9fba465b2ff44b5e29e190a4772c9fe65
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: smoke-gpu-interrupt-resume.txt
  locator: research/evidence/T-003/smoke-gpu-interrupt-resume.txt
  sha256: 59da83bc40d360260fd57ad6f1752903e10910f91f95f56bfad82be4212c0a28
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: test_sweep.py
  locator: research/tests/test_sweep.py
  sha256: 33b46ce045782810695f2f8dec61a2d02919a9a253207e7573c6b69f0f8a91b0
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-005
  kind: file
  label: runbook-walkthrough.txt
  locator: research/evidence/T-003/runbook-walkthrough.txt
  sha256: d5963af231841f4a411d0888aab8ed7c58957b93fadbf45f0aa715863599e544
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-006
  kind: file
  label: README.md
  locator: research/evidence/snapshots/README-at-T-003.md
  sha256: 0ce0ee80d56f4d5348a664c8dde5b38a5a208b132f58ae6642e532c90cfd638e
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: Sweep runner dispatches across both GPUs through the harness, resumes only interrupted
  configs, records failures, and collates reproducibly; runbook replayed from a fresh worktree.
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

- [x] AC-01: A multi-configuration sweep runs across both local GPUs through the harness
  Evidence: sweep log and collated table
- [x] AC-02: An interrupted sweep resumes without rerunning completed configurations and collation matches the uninterrupted table
  Evidence: resume log and table diff
- [x] AC-03: A failing configuration is reported as failed in collation rather than dropped
  Evidence: collated table row
- [x] AC-04: The runbook commands execute as written
  Evidence: runbook walkthrough output

## Evidence

- E-001: smoke-gpu-sweep.log — `research/evidence/T-003/smoke-gpu-sweep.log` (verified)
- E-002: smoke-gpu-table.csv — `research/evidence/T-003/smoke-gpu-table.csv` (verified)
- E-003: smoke-gpu-interrupt-resume.txt — `research/evidence/T-003/smoke-gpu-interrupt-resume.txt` (verified)
- E-004: test_sweep.py — `research/tests/test_sweep.py` (verified)
- E-005: runbook-walkthrough.txt — `research/evidence/T-003/runbook-walkthrough.txt` (verified)
- E-006: README.md — `research/README.md` (verified)

## Resolution

Sweep runner dispatches across both GPUs through the harness, resumes only interrupted configs, records failures, and collates reproducibly; runbook replayed from a fresh worktree.
