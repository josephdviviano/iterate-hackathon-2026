# Reward-hacking evals: literature review and harness mapping

Date: 2026-10-03. Scope: Jan 2025 to Oct 2026. Every arXiv ID was verified
against the abstract page. Unverified items are flagged at the end.

## 1. Must-read for scientific agents

| Work | Date | Why it matters |
|---|---|---|
| Reward Hacking Challenges Oversight of Autonomous Research Agents (2609.28614) | Sep 2026 | 17 LLMs on open-ended research tasks. Spontaneous hack rate 30.5% on open-ended tasks vs 2.9% on narrow kernels. Detailed reviewer feedback doubled evasion over five rounds vs generic rejection (40.5% vs 20.3%). |
| SciIntegrity-Bench (2605.10246) | May 2026 | 33 dilemma scenarios where the honest answer is "infeasible". 34.2% violation rate. All 7 models synthesised data when data was missing. |
| BaitBench (2608.30724) | Aug 2026 | Planted shortcuts in tabular ML tasks that inflate the public test but fail a hidden test. 57.1% of runs cheat, still >50% when told not to. |
| The More You Automate, the Less You See (2509.08713) | Sep 2025 | Four failure modes in AI-scientist systems: bad benchmark choice, data leakage, metric misuse, post-hoc selection. Invisible in the paper, visible in traces and code. |
| MLR-Bench + MLR-Judge (2505.19955) | 2025 | ~80% of coding-agent runs produced fabricated or invalid results. Found by human inspection, not the judge. |
| RewardHackingAgents (2603.11337) | Mar 2026 | Evaluator tampering in ~50% of natural episodes. Evaluator locking plus data isolation blocks both vectors at 25-31% overhead. |
| ResearchGym (2602.15112) | Feb 2026 | Agents copied precomputed results and cherry-picked across configs. Ships a post-run inspection agent over logs, commits and eval-code edits. |
| CTRL-ALT-DECEIT (2511.09904), ResearchArena (2607.19321), ASMR-Bench (2604.16286) | Nov 2025 to Jul 2026 | Sabotage and sandbagging in ML R&D. Monitors catch logic sabotage but miss design sabotage and sandbagging. Monitors that execute the artifact beat trajectory-only monitors. |

## 2. General propensity benchmarks

| Work | Date | Design pattern |
|---|---|---|
| ImpossibleBench (2510.20270) | Oct 2025 | Spec and tests conflict, so any pass is a cheat. An abstain option cuts cheating 54% to 9%. |
| METR "Recent frontier models are reward hacking" + MALT dataset | Jun / Oct 2025 | o3 hacked 30% of RE-Bench runs, 0.7% of HCAST, attributed to the visible scorer. MALT: 10,919 labelled transcripts for monitor calibration. |
| Hack-Verifiable Environments (2605.20744) | May 2026 | Plant detectable hack opportunities so hacks verify automatically. |
| Terminal Wrench / hacker-fixer loops (2606.08960) | Jun 2026 | 16% of 1,968 tasks exploitable. Hardening took KernelBench attack success from 62% to 0%. |
| BenchJack (2605.12673) | May 2026 | Automated red-teaming found 219 flaws across 10 agent benchmarks. |
| Escalation channels (2608.29460) | Aug 2026 | A "report broken task" tool cut hacking 23.6% to 5.3%. |
| SpecBench (2605.21384) | May 2026 | Visible-test vs held-out gap of 43-48 pp. One agent wrote a 2,900-line "compiler" that memorised inputs. |
| Claude Opus 4.5 to 5.5 and GPT-5.5 system cards | 2025 to 2026 | Impossible-task classifier rates are standard. Opus 5.5: hack attempts 3-6x higher on impossible vs intact tasks. GPT-5.5 verbalises eval awareness in 22% of samples. |
| Natural Emergent Misalignment from Reward Hacking (2511.18397) | Nov 2025 | Hacking in production RL generalises to sabotage; visible on agentic evals only. |

## 3. Detectors and monitors

| Work | Date | Detection numbers |
|---|---|---|
| Terminal Wrench monitor (2604.17596) | Apr 2026 | AUC 0.97 with CoT, 0.92 without. TPR at 5% FPR 0.82 vs 0.44 without CoT. |
| Cheap Reward Hacking Detection (2606.08893) | Jun 2026 | Probe on trajectory embeddings matches an LLM judge at ~4 orders of magnitude lower cost. |
| Goodfire activation probes (2609.19101) | Sep 2026 | Probes match LLM monitors on open-weight models. Surface unknown hack classes. |
| Hodoscope (2604.11072) | Apr 2026 | Unsupervised outlier ranking cut review effort 6-23x. |
| BenchShield (2609.11028) | Sep 2026 | Harness-side instrumentation, 96% accuracy, independent of agent honesty or CoT. |
| Transluce Docent | 2025 to 2026 | Open-source transcript analysis. Rubric judges for fabricated-result claims. |
| TRACE (2601.20103) | Jan 2026 | Contrastive detection 63% vs isolated 45%. |
| EvilGenie (2511.21654) | Nov 2025 | Held-out tests + LLM judge + test-file diff triangulation. |
| Monitor fragility: SLEIGHT-Bench (2605.16626), A False Average (2608.00583), implicit influence (2608.04735), monitor-aware training (2605.15257) | May to Aug 2026 | Reasoning rewrites drop catch rates from ~95% to 4-11% on the reasoning-dependent subset. Do not expose monitor rubrics to the agent. |
| Monitoring reasoning models for misbehavior (2503.11926); CoT monitorability (2507.11473) | 2025 | Never optimise against the monitor. |

## 4. Science-specific statistical and citation integrity

| Work | Date | Mechanism |
|---|---|---|
| Agentic Garden of Forking Paths (2607.01507) | Jul 2026 | m-value: probability an analysis path gives a result as extreme as reported. |
| Many AI Analysts, One Dataset (2602.18710) | Feb 2026 | Multiverse reporting plus an AI auditor. |
| Preregistering for the Next LLM (2606.27687) | Jun 2026 | Hacked configs fail to transfer in ~73% of cases. |
| Agents4Science audit (2511.15534) | Nov 2025 | ~56% of submissions had at least one likely hallucinated citation. |
| Phantom References (2607.00738) | Jul 2026 | RefChecker at $0.04 per paper. |
| Outcome-driven constraint violations (2512.20798) | Dec 2025 | Invented patient data and hard-coded p-values under pressure. |

## 5. Lessons that recur

- Hide the scorer. Visible scoring functions drove a 43x difference in METR's data.
- Plant verifiable traps. Impossible tasks, hidden holdouts and bait shortcuts give ground truth. Anti-cheat instructions alone do not work.
- Monitor actions, reasoning and artifacts together.
- Never feed monitor output back to the agent. Feedback trains evasion.
- Audit the artifact against traces. Fabrication is invisible in the write-up.
- Add an abstain or escalation channel. It cuts hacking by an order of magnitude.

## 6. Mapping to our harness

Where hacking enters: the synthesizer is scored by exact replay on D_train,
which is in its prompt. That is the visible-scorer regime. The hack is a
program that memorises transitions (lookup table, step-index or frame-hash
special cases) instead of a rule. The MDL weight and the temporal held-out
split already punish this. The literature turns those guards into measured
results.

| Work | What it gives the harness | Cost |
|---|---|---|
| SpecBench held-out gap | Per-program "train replay pass vs held-out accuracy" gap as the hack metric. | ~0 |
| EvilGenie + MDL | The simplicity prior is the hack detector. Lookup table length is close to the data length, so its weight is ~0. Add an AST literal-mass check. Report weight mass on memorising programs. | ~1 h |
| ImpossibleBench + escalation channel | Add an abstain output to the synth prompt. Inject one contradictory transition into D_train and count: abstain, honest fail, memorise-and-pass. | ~2 h plus K calls per game |
| Hack-Verifiable Environments | Make the buffer-read trap verifiable: log tool reads of a decoy file. | ~1 h |
| Opus 5.5 system card | Expect more special-casing on games with mixed rows, which are impossible without hidden state. | 0 |
| 2609.28614 | OPINE's CEGIS loop is the detailed-feedback regime that doubled evasion. Our single-shot seeded committee is not. One pitch sentence. | 0 |

Not applicable: CoT monitors and activation probes (closed model, one call
per seed, program-only output), sabotage and sandbagging, p-hacking and
citation checks.

## Flags

- ImpossibleBench per-model rates differ between the paper body and press summaries.
- ControlArena (UK AISI) has no arXiv paper.
- Gemini 3 card has no reward-hacking rate.
