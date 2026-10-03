# BioProt Selective-Prediction Eval — Implementation Handoff

Oct 3, 2026 · @Joseph Viviano

## Objective

Measure whether an agent's own uncertainty on BioProt protocol generation predicts protocol quality well enough to gate execution before a bad plan reaches a robot or a bench.

The eval does not ask how good the generated protocols are. It asks whether the agent knows which of its own protocols are bad. Those are different quantities, and only the second one supports a short-circuit.

**Success criterion.** An uncertainty signal is useful if the area under the risk-coverage curve (AURC) is materially below the random-ordering baseline, and selective risk at 50% coverage is at least a third lower than risk at full coverage. Anything that only matches random ordering is a null result and should be reported as one.

**Out of scope.** Improving protocol quality, tuning prompts for higher accuracy, and wet-lab execution. Do not optimise generation quality mid-study; it changes the risk distribution the curves are computed over.

## Data

Primary corpus is BioProt, from the BioPlanner paper ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.162), [arXiv](https://arxiv.org/abs/2310.10632), code and data at [github.com/bioplanner/bioplanner](https://github.com/bioplanner/bioplanner)). BIOPROT 2.0 ships with ProtocoLLM / ProtoMed-LLM ([arXiv](https://arxiv.org/abs/2410.04601), [github.com/ProtocoL-LLM/ProtocoLLM](https://github.com/ProtocoL-LLM/ProtocoLLM.git)).

| Field | BioProt | BIOPROT 2.0 |
| --- | --- | --- |
| Protocols | 100 | larger; confirm count from the repo before planning around it |
| Avg steps per protocol | 12.5 | unconfirmed |
| Avg pseudocode lines | 17.2 | unconfirmed |
| Avg pseudofunctions per protocol | 10.3 | n/a — fixed action set |
| Action space | generated per protocol by GPT-4 | predefined lab actions shared across protocols |
| Source | protocols.io, filtered from 9,000+ public protocols | protocols.io, keyword-scored selection |

**Pool them as strata, not as one set.** The action-space difference in the last two rows is the reason. BioProt gives the model a protocol-specific function list; BIOPROT 2.0 gives a shared predefined vocabulary. Task difficulty and the meaning of a wrong function call differ, so compute curves separately and only merge if the per-stratum curves agree.

**Contamination.** Both corpora have been public since 2023 and 2024 and are built from open protocols.io content, so assume partial memorisation. This inflates absolute accuracy but is mostly benign for this study, which measures ranking within a model's own outputs. It is not benign for one thing: memorised protocols may draw high confidence for reasons unrelated to reasoning. Flag any protocol the model reproduces near-verbatim.

**Open question for the implementer.** Confirm BIOPROT 2.0 ships ground-truth pseudocode in a parseable format matching BioProt's. If it does not, drop it and run on BioProt alone rather than writing a converter.

## Task setup

Run **full protocol generation**, not next-step prediction. Next-step prediction scores a single choice with no plan to gate; protocol generation produces the whole executable artifact, which is the thing a short-circuit would stop.

**Inputs per item.** Protocol title, protocol description, and the admissible pseudofunction set. The model returns pseudocode using only those functions.

**Fixed conditions for the main run:**

1. **Shuffle the pseudofunctions.** Unshuffled, the functions appear roughly in call order because they were generated sequentially, which leaks the answer. BioPlanner reports a large performance drop when shuffled, confirming the leak. Shuffled is the honest condition.
2. **Feedback loop off.** The error-checking loop catches undefined functions and Python syntax errors. Leave it off in the main condition: it changes the artifact being judged and partially does the job the uncertainty signal is supposed to do. Run it as an ablation instead.
3. **Use the original human descriptions as primary.** The dataset also ships GPT-4-generated descriptions, which score slightly higher. Report both; the generated ones are a separate condition, not the default.
4. **Temperature > 0, k samples per protocol.** Needed for the self-consistency signal in the next section. BioPlanner reported over 5 runs; match that as a floor.

**Record the raw generation verbatim** alongside every derived metric. Re-running generation to recover something you did not store is the most likely way to burn the budget twice.

## Uncertainty elicitation

Collect **all four signals per item in the same run** and store them side by side. Comparing them later must not require regenerating anything.

| # | Signal | How | Why it ranks here |
| --- | --- | --- | --- |
| 1 | Self-consistency | k samples per protocol; score = mean pairwise normalised Levenshtein distance between the model's own function sequences | Label-free, no prompt dependence, and the dataset's own metric does the comparing. Default primary. |
| 2 | Verbalised confidence + abstain | 0–100 confidence, plus an explicit "insufficient information to write this protocol safely" option | Mirrors LAB-Bench's abstain design, so results are comparable to the one biology benchmark that scores abstention |
| 3 | Sequence logprob | Mean token logprob over the generated pseudocode, length-normalised | Cheap where the API exposes it; unavailable for some models, so it cannot be primary |
| 4 | Self-critique | Second pass: show the model its own protocol, ask for a pass/fail execution judgement | Weakest prior — BioPlanner found GPT-4 only slightly above chance at distinguishing ground-truth from generated pseudocode |

**Signal 1 is the default headline.** Signals 2 and 4 both route through the model's self-assessment, which the BioPlanner authors already showed to be near chance on this exact material. Treat a strong result from 2 or 4 as surprising and check it before believing it.

**Keep elicitation out of the generation prompt.** Ask for confidence in a separate call against the stored generation, so the confidence request cannot change the protocol being judged.

**Record the abstain rate separately from the confidence score.** They answer different questions and a model that never abstains still produces a usable confidence ranking.

## Correctness labels

Primary risk is **normalised Levenshtein distance between the predicted and ground-truth function sequences**, treating each function call as one symbol, divided by the number of ground-truth functions. Lower is better. BioPlanner's own harness computes it, so reuse that code rather than reimplementing.

Secondary metrics, all reported, none used for the headline curve: function precision and recall (counting repeated calls), argument name precision and recall, SciBERTScore over argument values, BLEU over argument values.

**Binarisation.** For the headline risk-coverage curve, call a protocol acceptable at normalised Levenshtein ≤ 0.4. That sits just above GPT-4's best reported score of 0.396 in the original paper, so roughly half of a strong model's outputs land on each side, which is where a coverage curve has the most resolution. Preregister the threshold before looking at results, and also report the curve with continuous risk so the conclusion does not hinge on the cut.

**Do not use an LLM judge as the primary label.** BioPlanner tested exactly this and found GPT-4 preferred the model's generation over ground truth 40–44% of the time, barely distinguishable from chance. An LLM judge is acceptable only as a reported secondary.

**Hand-label a severity subset.** Sample 20 protocols spanning the risk range and have someone with wet-lab background mark each as: correct, cosmetically wrong, or wrong in a way that would waste reagents or produce a dangerous step. The edit distance cannot tell these apart, and the third category is the only one that justifies a gate. The severity subset is what tells you whether the curve means anything operationally.

## Metrics

Sort items by uncertainty ascending, then sweep the abstention threshold from full coverage down to zero.

| Metric | Definition | Role |
| --- | --- | --- |
| Risk-coverage curve | Selective risk among retained items, against fraction retained | The headline figure |
| AURC | Area under that curve | Single-number comparison across signals and models |
| Selective risk at 0.9 / 0.75 / 0.5 coverage | Risk among the most-confident 90%, 75%, 50% | What a real gate would deliver at three operating points |
| Coverage at fixed risk | Fraction retainable while holding risk at 0.1 | The dual question: how much work survives a safety bar |
| Precision and coverage | Correct ÷ attempted, attempted ÷ total, for the explicit-abstain condition only | Comparable to LAB-Bench |
| ECE and Brier | Over verbalised confidence | Calibration, distinct from ranking quality |

**Two baselines are mandatory on every plot.** Random ordering, whose curve is flat at the full-coverage risk, and oracle ordering by true risk, which is the achievable floor. A signal's value is where it sits between them. Reporting AURC without the random baseline makes any number look meaningful.

**Bootstrap over protocols, never over samples.** The k samples of one protocol are not independent observations. Resample protocol ids with replacement, 1,000 draws, and report 95% intervals on AURC and on each selective-risk point.

**Report the tie fraction** for verbalised confidence. It tends to cluster on round numbers, and heavy ties make the curve depend on tie-breaking. Break ties deterministically by protocol id and say so.

## Experiment matrix

Main run: three models from different families × 100 BioProt protocols × k = 5 samples = 1,500 generations, plus one confidence call per generation. Add BIOPROT 2.0 as a second stratum if the format check in Data passes.

Ablations, each against the same models and protocols:

| Ablation | Contrast | What it tests |
| --- | --- | --- |
| Function order | shuffled vs unshuffled | Uncertainty should rise when the order leak is removed. If it does not, the signal is not tracking task difficulty. |
| Description source | human vs GPT-4-generated | Known to shift performance; checks the curve is not an artefact of description quality |
| Feedback loop | off vs on | Does a syntax-error loop already capture what the uncertainty signal captures? |
| k | 3 vs 5 vs 10 | Sample budget needed for a stable self-consistency score |

**Power is the binding constraint.** With 100 protocols, the 50%-coverage point rests on 50 items and confidence intervals will be wide. Plan for this rather than discovering it: do not report any difference between two signals or two models whose bootstrap intervals overlap, and prefer within-model comparisons of signals, which are paired and therefore tighter, over cross-model claims.

**Fix seeds and record them** per generation. The whole study is a ranking over a specific set of outputs; an unreproducible generation set makes every number unauditable.

## Pitfalls

**Semantically identical functions score as errors.** BioPlanner names this directly: `Mix` and `MixSubstance` mean the same thing but differ syntactically, and the model is penalised for picking the wrong one. Edit distance will read legitimate paraphrase as risk. Manually inspect the 20 highest-risk items before trusting the top of the curve.

**Syntactic validity is not assay correctness.** A protocol can use every function correctly and still be biologically wrong — wrong buffer, wrong incubation, wrong order of reagents. An uncertainty signal that tracks only surface errors will look well-calibrated against edit distance while missing exactly the failures a gate exists to catch. The severity subset in Correctness labels is the check for this; run it before writing conclusions, not after.

**Memorisation inflates confidence.** A protocol reproduced from training data draws high confidence for a reason that will not transfer to a novel protocol. Flag near-verbatim reproductions and report the curve with and without them.

**Self-assessment signals have a known ceiling here.** GPT-4 scored 40–44% at telling its own generation from ground truth. Do not build the headline result on signals 2 or 4 without a sanity check.

**Argument metrics are computed only on correctly predicted functions** in the original harness, so they are conditional, not marginal. Do not average them across items with different function accuracy and present the result as an overall score.

**The model may abstain for the wrong reason** — safety-flavoured refusal on a protocol it could write fine. Separate "declined" from "uncertain" in the logs; they are different events and only one is a calibration signal.

## Deliverables

Build in this order. Each step gates the next.

1. **Reproduction gate.** Run BioPlanner's own protocol-generation harness on one model and reproduce its reported numbers within noise — normalised Levenshtein around 0.4 to 0.7 depending on shuffle, function precision in the low 90s. If this does not reproduce, stop and fix it. Every later number inherits this harness.
2. **Generation run.** `generations.jsonl`, one row per sample: `protocol_id`, `stratum`, `model`, `seed`, `sample_idx`, `prompt_hash`, `raw_output`, `parsed_function_sequence`, condition flags for shuffle / description source / feedback.
3. **Scoring pass.** `scores.jsonl`, joined on the same keys: every BioProt metric plus the binarised acceptable flag, with the threshold recorded in the file.
4. **Uncertainty pass.** `uncertainty.jsonl`: all four signals per sample, plus `declined` and `abstained` as separate booleans.
5. **Metrics module.** Pure functions from (risk, uncertainty) pairs to risk-coverage curves, AURC, selective risk, with the random and oracle baselines computed in the same call. No notebook state.
6. **Severity subset.** 20 hand-labelled protocols, stored as `severity.csv` with the labeller's notes.
7. **Writeup.** The curves, the baselines, the severity cross-check, and a plain statement of whether the success criterion in Objective was met.

Scripts, not notebooks. Every output file self-describing enough to re-run analysis without the code that produced it.

**Report a null result as a null result.** The most likely outcome is that these signals rank only slightly better than random, and that is worth knowing before anyone builds a gate on them.
