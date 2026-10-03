---
schema_version: '2.0'
task_id: T-014
title: Converge the climbed recipe into submission defaults
work_type: delivery
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
objective: Make the team folder's defaults run the T-013 converged recipe (D-005 to D-007) without parameters,
  apply compile and fused SGD only on CUDA, remove exploration-only code paths that the recipe does not
  use, write the submission README, and run one simplify-codebase pass over the integrated team folder.
consequence: The upstream PR must run its final recipe from defaults; exploration code left in place enlarges
  the compliance review surface.
decision_effect: Produces the exact candidate for A100 qualification and compliance assurance.
uncertainty: null
dependencies:
- T-013
status: completed
acceptance:
- criterion_id: AC-01
  statement: Default settings reproduce the converged recipe's dev-stack accuracy within noise over 10
    fresh seeds
  expected_evidence: harness summary
  status: verified
  evidence_ids:
  - E-001
  - E-002
  - E-003
  - E-004
  rationale: null
- criterion_id: AC-02
  statement: Contract tests pass on the pinned stack and the team folder contains only source and README
  expected_evidence: pytest output and diff inspection
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
  label: s20-defaults-table.csv
  locator: research/evidence/T-014/s20-defaults-table.csv
  sha256: 5ee384917d7f8f0702465076790b1c88bb46b6643ff817b9cbad43519121589d
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-002
  kind: file
  label: s20-defaults.toml
  locator: research/evidence/T-014/s20-defaults.toml
  sha256: ddd0c2ec9efde77361e2938ba7bd8f545604450698fccdbff8139ed92c7eaf85
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-003
  kind: file
  label: checks.txt
  locator: research/evidence/T-014/checks.txt
  sha256: 58963fb86f12829b824d61f30cd92cb1c7ed8330cabf2738b1ab673018dcd155
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
- evidence_id: E-004
  kind: file
  label: test_submission-at-T-014.py
  locator: research/evidence/T-014/test_submission-at-T-014.py
  sha256: 46e5485f6e249771d41125cb34f6233e68024b1c11ca189e38d255d416330394
  state: verified
  candidate_identity: null
  obligation_ids: []
  note: null
completion_summary: 'Team folder converged and simplified: defaults are the T-013 recipe (75.21% over
  10 fresh seeds at 5.18 s local), exploration paths moved to research/lab_recipe, training bit-identical
  to the lab substrate, compile and fused SGD CUDA-only, README written; 23 contract tests pass on the
  pinned stack.'
blocker_ids: []
---

# T-014: Converge the climbed recipe into submission defaults

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Make the team folder's defaults run the T-013 converged recipe (D-005 to D-007) without parameters, apply compile and fused SGD only on CUDA, remove exploration-only code paths that the recipe does not use, write the submission README, and run one simplify-codebase pass over the integrated team folder.

## Consequence

The upstream PR must run its final recipe from defaults; exploration code left in place enlarges the compliance review surface.

## Decision effect

Produces the exact candidate for A100 qualification and compliance assurance.

## Acceptance

- [x] AC-01: Default settings reproduce the converged recipe's dev-stack accuracy within noise over 10 fresh seeds
  Evidence: harness summary
- [x] AC-02: Contract tests pass on the pinned stack and the team folder contains only source and README
  Evidence: pytest output and diff inspection

## Evidence

- E-001: s20-defaults-table.csv — `research/evidence/T-014/s20-defaults-table.csv` (verified)
- E-002: s20-defaults.toml — `research/evidence/T-014/s20-defaults.toml` (verified)
- E-003: checks.txt — `research/evidence/T-014/checks.txt` (verified)
- E-004: test_submission-at-T-014.py — `research/evidence/T-014/test_submission-at-T-014.py` (verified)

## Resolution

Team folder converged and simplified: defaults are the T-013 recipe (75.21% over 10 fresh seeds at 5.18 s local), exploration paths moved to research/lab_recipe, training bit-identical to the lab substrate, compile and fused SGD CUDA-only, README written; 23 contract tests pass on the pinned stack.
