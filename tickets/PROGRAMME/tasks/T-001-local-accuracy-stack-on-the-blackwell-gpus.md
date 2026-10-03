---
schema_version: '2.0'
task_id: T-001
title: Local accuracy stack on the Blackwell GPUs
work_type: delivery
commitment: required
workset_id: enablement
requirement_ids:
- R-006
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- exploration-tooling
review_path_ids:
- decision-review
objective: Provide a reproducible dev stack (torch 2.7.1+cu128) beside the pinned stack, with CIFAR-100
  downloaded, so the official harness runs real-data trials on both local GPUs.
consequence: 'Without it no local probe can run: the pinned torch 2.4.0 has no sm_120 kernels.'
decision_effect: Enables every local accuracy probe; does not license timing claims.
uncertainty: null
dependencies: []
status: completed
acceptance:
- criterion_id: AC-01
  statement: The official harness completes a real-data CUDA trial on each local GPU with the dev stack
  expected_evidence: harness summary.json per GPU
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
- criterion_id: AC-02
  statement: The dev stack is reproducible from a recorded freeze and setup commands
  expected_evidence: research/env freeze file and runbook section
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
- Dev-stack accuracy may differ from the pinned A100 stack; T-004 owns that check (R-007).
evidence:
- evidence_id: E-001
  kind: file
  label: real-data-trials.txt
  locator: research/evidence/T-001/real-data-trials.txt
  sha256: b2cd37f55d9b0609f9cd5ba6a04461d2d1ab6cb6c03b7ce112fb27bb3639fe48
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: blackwell-freeze.txt
  locator: research/env/blackwell-freeze.txt
  sha256: 0d42c40c493d7f4cd7956e524b8c4f989c691e510980c93a9011000fecb4dd98
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: setup-blackwell.sh
  locator: research/env/setup-blackwell.sh
  sha256: 958457c750d9e631aad309e57f8a7c4c63ffa1efe2ef52fce18e1e792e66de13
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: runbook-walkthrough.txt
  locator: research/evidence/T-003/runbook-walkthrough.txt
  sha256: d5963af231841f4a411d0888aab8ed7c58957b93fadbf45f0aa715863599e544
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: Dev stack torch 2.7.1+cu128 runs the official harness on real data on both local GPUs;
  setup script and freeze reproduce it from a fresh worktree.
blocker_ids: []
---

# T-001: Local accuracy stack on the Blackwell GPUs

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Provide a reproducible dev stack (torch 2.7.1+cu128) beside the pinned stack, with CIFAR-100 downloaded, so the official harness runs real-data trials on both local GPUs.

## Consequence

Without it no local probe can run: the pinned torch 2.4.0 has no sm_120 kernels.

## Decision effect

Enables every local accuracy probe; does not license timing claims.

## Acceptance

- [x] AC-01: The official harness completes a real-data CUDA trial on each local GPU with the dev stack
  Evidence: harness summary.json per GPU
- [x] AC-02: The dev stack is reproducible from a recorded freeze and setup commands
  Evidence: research/env freeze file and runbook section

## Risks

- Dev-stack accuracy may differ from the pinned A100 stack; T-004 owns that check (R-007).

## Evidence

- E-001: real-data-trials.txt — `research/evidence/T-001/real-data-trials.txt` (verified)
- E-002: blackwell-freeze.txt — `research/env/blackwell-freeze.txt` (verified)
- E-003: setup-blackwell.sh — `research/env/setup-blackwell.sh` (verified)
- E-004: runbook-walkthrough.txt — `research/evidence/T-003/runbook-walkthrough.txt` (verified)

## Resolution

Dev stack torch 2.7.1+cu128 runs the official harness on real data on both local GPUs; setup script and freeze reproduce it from a fresh worktree.
