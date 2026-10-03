---
schema_version: '2.0'
task_id: T-006
title: Select the base regime
work_type: decision
commitment: required
workset_id: frontier
requirement_ids:
- R-003
scenario_ids:
- S-001
assumption_ids: []
affected_boundaries:
- submission-folder
review_path_ids: []
objective: Choose architecture, width, depth, batch size and epoch count by minimum interpolated A100
  PCIe time at a 75.3% single-view mean, steelmanning the runner-up regime.
consequence: Every add-on comparison and the final recipe inherit this base.
decision_effect: Fixes the base for the add-ons workset and states reopening conditions.
uncertainty: null
dependencies:
- T-004
- T-005
status: completed
acceptance:
- criterion_id: AC-01
  statement: A decision record selects the regime from the frontier table and A100 timings, with rejected
    alternatives and reopening conditions
  expected_evidence: decision record
  status: verified
  evidence_ids:
  - E-001
  rationale: null
evidence_requirements: []
out_of_scope: []
risks: []
evidence:
- evidence_id: E-001
  kind: file
  label: D-008-which-base-regime-does-the-entry-use-given-the-frontier-and-a100.yaml
  locator: tickets/PROGRAMME/decisions/D-008-which-base-regime-does-the-entry-use-given-the-frontier-and-a100.yaml
  sha256: d528ccc4233ee11917ecb21ba8bf6c28852f2efb9edde354ec034ee6993a48c4
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: Regime settled on 128/384/640 at 8.25 epochs from the frontier and A100 timings (D-008).
blocker_ids: []
---

# T-006: Select the base regime

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Choose architecture, width, depth, batch size and epoch count by minimum interpolated A100 PCIe time at a 75.3% single-view mean, steelmanning the runner-up regime.

## Consequence

Every add-on comparison and the final recipe inherit this base.

## Decision effect

Fixes the base for the add-ons workset and states reopening conditions.

## Acceptance

- [x] AC-01: A decision record selects the regime from the frontier table and A100 timings, with rejected alternatives and reopening conditions
  Evidence: decision record

## Evidence

- E-001: D-008-which-base-regime-does-the-entry-use-given-the-frontier-and-a100.yaml — `tickets/PROGRAMME/decisions/D-008-which-base-regime-does-the-entry-use-given-the-frontier-and-a100.yaml` (verified)

## Resolution

Regime settled on 128/384/640 at 8.25 epochs from the frontier and A100 timings (D-008).
