# Related methods and baselines for the program committee

Compiled 2026-10-03 from three focused reviews (learned dynamics models and
disagreement-driven exploration; program-synthesis world models and program
selection; uncertainty estimation for language-model outputs). Only
arXiv-identified works are listed.

## Closest prior work, and the exact difference

| Work | Relation to the committee |
|---|---|
| Query by committee, Seung, Opper, Sompolinsky 1992 | Same idea: keep hypotheses consistent with the data, query where they disagree. Ours is QBC with LLM-synthesized programs, an exact-replay filter and a gzip-length prior. |
| Distinguishing inputs, Jha et al. 2010 (OGIS); Ji et al. 2020 (PLDI) | Our probe rule: among consistent programs, query the input where they differ. We add the same disagreement as a calibrated error predictor on held-out transitions. |
| Semantic entropy, Kuhn, Gal, Farquhar 2023 (arXiv:2302.09664); Farquhar et al., Nature 2024 | Our disagreement is semantic entropy with equivalence classes defined by execution (identical predicted next state) instead of entailment. |
| AlphaCode, Li et al. 2022 (arXiv:2203.07814); MBR-Exec (arXiv:2204.11454); CodeT (arXiv:2207.10397) | Grouping programs by identical outputs and taking the largest group is their selection rule; we take the entropy over the groups per input. Keeping only train-exact programs is CodeT's execution agreement with real tests. |
| Functional entropy for code, Bouchard et al. 2026 (arXiv:2605.28500) | Nearest prior art: semantic entropy with functional-equivalence clusters, aggregated per task. We score per transition, weight by MDL and gate on exact replay. |
| Self-consistency, Wang et al. 2022 (arXiv:2203.11171) | Majority over temperature samples. Our unseeded control is this with a verifier. |
| Hypothesis Search, Wang et al. 2023 (arXiv:2309.05660) | Natural-language hypotheses implemented as programs and filtered on examples. Our seed hypotheses are this step; they keep one passer, we keep all and vote. |
| OPINE-World, Courtis, Li, Sanner 2026 (arXiv:2607.01531) | Our data and verifier. One CEGIS-repaired program; exploration by Dirichlet entropy of effect counts. We keep a committee and replace the count entropy by disagreement, defined also on rows with no counts. |
| WorldCoder (arXiv:2402.12275); Code World Models (arXiv:2405.15383); EMPA (arXiv:2107.12544); Rodionov 2026 (arXiv:2605.05138) | Single or MAP program world models, some with a simplicity prompt. No committee, no disagreement. |
| PoE-World, Piriyakulkij et al. 2025 (arXiv:2505.10819) | Product of many partial expert programs with fitted weights. Whole programs, plurality vote and MDL weights here; no uncertainty use there. |
| Disagreement exploration, Pathak et al. 2019 (arXiv:1906.04161); MAX (arXiv:1810.12162); Plan2Explore (arXiv:2005.05960); deep ensembles (arXiv:1612.01474); BALD (arXiv:1112.5745) | Ensemble disagreement as epistemic uncertainty and exploration reward, with neural members. Our members are programs filtered by exact replay; BALD equals vote entropy for deterministic members. |
| DreamCoder (arXiv:2006.08381) | MDL prior over programs; ours is gzip length of the stripped source. |

## What is not new, and what is

Not new on its own: sampling programs and filtering on examples; natural-language
hypotheses before code; grouping programs by behaviour; a length prior;
disagreement as an exploration signal; program ensembles as world models.

New, as a measurement: version-space disagreement over exact-replay-verified
transition programs is a calibrated error signal on ARC-AGI-3 object data
(unanimous error 0.16 to 0.30, split error 0.67 to 1.00, AUROC 0.68 to 0.78),
it is defined on rows where count-based uncertainty is undefined, and it gives
a falsifying probe in one step against 7 to 10 for count priority. State it
as query by committee applied to programmatic world models.

## Baselines a reviewer expects, and their status

| Baseline | Why | Status |
|---|---|---|
| Single program, count-priority exploration (OPINE-World) | The method being extended | Done, R3, R4, R6 |
| Unseeded self-consistency: same synthesizer, no seed hypotheses, same verifier and weights | Isolates the seeds from the execution-defined entropy | Running, `unseeded_devin` |
| Nearest-neighbour effect copy | Floor with no uncertainty | Done, `committee.baselines` |
| Bagged decision trees (classic QBC) and MLP deep ensemble on per-object features | The dynamics-ensemble baseline for disagreement | Done, `committee.baselines` |
| Verbalized confidence of a judge model over the plurality program's prediction | The LLM-uncertainty baseline (Kadavath 2022, Tian 2023) | Running, `committee.confidence` |
| Uniform vs MDL weights; shortest program alone; largest cluster alone | Ablations of the weighting | Done in part (λ sweep, R3) |
| No verifier, soft weights by train accuracy | Is exact replay load-bearing | Not run; few inconsistent programs exist |
| K sweep with bootstrap confidence intervals, pooled over levels | Small n: 44 transitions per level gives AUROC an interval near 0.1 | To do from stored members |

## Metrics to report beyond AUROC

Brier and ECE with the plurality weight share as the probability (at most 5
equal-mass bins at n=44; ECE is biased upward at small n); selective accuracy
at 50% and 80% coverage and AURC (Geifman et al. 2019); bootstrap confidence
intervals clustered by level, since transitions within a level share a
program's errors. The disagreement score takes few distinct values with 8
members, so AUROC has ties; risk-coverage curves handle that.

## Strongest critique

The eight members come from one synthesizer and one prompt, so disagreement
measures sampling variance, and agreement can hide a shared wrong bias; the
agreed-but-wrong rate of 16 to 30% shows this. The answer is the K sweep and
the seeded-versus-unseeded comparison with confidence intervals, and the
ensemble baseline on the same split. Transition-level accuracy is also not
play: arXiv:2607.14169 reports perfect transition accuracy coexisting with
losing games.
