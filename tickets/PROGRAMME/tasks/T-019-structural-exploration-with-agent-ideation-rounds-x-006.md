---
schema_version: '2.0'
task_id: T-019
title: Structural exploration with agent ideation rounds (X-006)
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
objective: 'Explore structural vectors beyond the converged recipe: ensembles within the untimed evaluation
  budget, closed-form head refit, schedule shapes, lookahead dynamics, data ordering, head geometry, in-run
  soft targets, pooling geometry and A100 systems levers; run agent ideation at the start and at each
  convergence point, and adopt only levers confirmed on fresh seeds and same-host A100 timing.'
consequence: Remaining gains, if any, lie outside the parameter space already climbed.
decision_effect: Adopts confirmed levers into the submission or establishes saturation of the structural
  space.
uncertainty: Whether any structural lever lowers A100 PCIe time to a 75.2% single-view mean beyond seed
  and host noise.
dependencies:
- T-018
status: in_progress
acceptance:
- criterion_id: AC-01
  statement: Each ideation round's hypotheses are recorded in exploration portfolio X-006 with a disposition
  expected_evidence: programme explore-check output
  status: pending
  evidence_ids: []
  rationale: null
- criterion_id: AC-02
  statement: Adopted levers are confirmed on fresh seeds and timed on A100 PCIe; others are rejected with
    evidence
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

# T-019: Structural exploration with agent ideation rounds (X-006)

> Canonical state is the YAML frontmatter. Use `programme` commands to update it.

## Objective

Explore structural vectors beyond the converged recipe: ensembles within the untimed evaluation budget, closed-form head refit, schedule shapes, lookahead dynamics, data ordering, head geometry, in-run soft targets, pooling geometry and A100 systems levers; run agent ideation at the start and at each convergence point, and adopt only levers confirmed on fresh seeds and same-host A100 timing.

## Consequence

Remaining gains, if any, lie outside the parameter space already climbed.

## Decision effect

Adopts confirmed levers into the submission or establishes saturation of the structural space.

## Acceptance

- [ ] AC-01: Each ideation round's hypotheses are recorded in exploration portfolio X-006 with a disposition
  Evidence: programme explore-check output
- [ ] AC-02: Adopted levers are confirmed on fresh seeds and timed on A100 PCIe; others are rejected with evidence
  Evidence: collated tables and findings
