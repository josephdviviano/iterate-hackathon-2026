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
status: in_progress
acceptance:
- criterion_id: AC-01
  statement: The official harness completes a real-data CUDA trial on each local GPU with the dev stack
  expected_evidence: harness summary.json per GPU
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: The dev stack is reproducible from a recorded freeze and setup commands
  expected_evidence: research/env freeze file and runbook section
  status: pending
  evidence_ids: []
  rationale: null
evidence_requirements: []
out_of_scope: []
risks:
- Dev-stack accuracy may differ from the pinned A100 stack; T-004 owns that check (R-007).
evidence: []
completion_summary: null
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

- [ ] AC-01: The official harness completes a real-data CUDA trial on each local GPU with the dev stack
  Evidence: harness summary.json per GPU
- [ ] AC-02: The dev stack is reproducible from a recorded freeze and setup commands
  Evidence: research/env freeze file and runbook section

## Risks

- Dev-stack accuracy may differ from the pinned A100 stack; T-004 owns that check (R-007).
