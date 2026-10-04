# Uncertainty committee

Calibration: 4 results scored; consensus range hit 50% (target ~80%); experimenter's sealed range hit 75%; counterexamples 2; disagreement predicts misses: AUROC 0.00; improve votes pick keeps: AUROC 1.00; recent bias +0.48 half-ranges; veto inactive until 5 results are scored.

## Members
- W1 active (mechanism, claude-opus-5-5, since E000): hit 50% of 4, recent 50%
- W2 active (empirical, claude-sonnet-5-5, since E000): hit 50% of 4, recent 50%
- W3 active (cost, claude-opus-5-5, since E000): hit 50% of 4, recent 50%
- W4 active (skeptic, claude-sonnet-5-5, since E000): hit 50% of 4, recent 50%

## Queued ideas
idea | median range | conflict | improve votes | harm votes | verdict entropy
---|---|---|---|---|---
I008 | [7.25, 8.149999999999999, 9.7] | 0.0 | 0/4 | 0/4 | 0.0

## Counterexamples
- E003 (I003): On the port, compile the model's forward+backward with torch.compile (mode='max-autotune-no-cudagraphs') in build. Compile and warm up on synthetic inputs for the train batch shape and the ragged last -> measured time=12.067211381334346, accuracy=0.7486333333333334, outside every member's range
- E005 (I004): On the port at 9 epochs, add Lookahead/EMA of weights in the airbench96_faster style: update every 5 steps, alpha ramping as in the reference, final weights set to the EMA copy. All EMA state is creat -> measured time=9.097539043003053, accuracy=0.7583000000000001, outside every member's range

## Cruxes (latest round)
- W1: Is the +0.97 pp from EMA (E005) real or partly seed luck? A re-run, or a run at about 9.3 epochs, decides how far epochs can be cut from 10.
- W1: Does warming up several input resolutions make torch.compile switch to dynamic shapes, slowing every step or recompiling inside timed code?
- W1: What fraction of the 9.1 s is fixed cost (prepare, first step) versus per-epoch cost? This decides whether shrinking prepare or cutting epochs and FLOPs is the bigger lever.
- W2: Does the compiled run recompile for each new resolution inside timed code, and how much does the compile warmup in build actually cover?
- W2: What was E004's epoch count, and how large is the true EMA gain at fewer epochs? This sets how many epochs can be cut.
- W2: What is the per-seed accuracy std? It decides whether a 0.5 pp margin survives the 40-seed judgment.
- W3: With EMA on, what is the accuracy slope from 10 down to 9-9.5 epochs, and can about 0.5 pp of margin be spent for about 0.5-0.9 s?
- W3: Does the compiled train step recompile or switch to dynamic shapes when given multiple input resolutions, even after warmup in build?
- W3: What are the per-seed std and the 40-seed mean offset from the 3-seed mean, which set how much margin above 0.753 must be kept?
- W4: Does the EMA gain (+0.97 pp vs E004) hold on re-run, or is it partly seed luck? The 3-seed std is about 0.15 pp. Margin over 0.753 is only 0.5 pp.
- W4: Is the per-step time compute-bound? If so, resolution reduction buys real time. Eager interpolate/crop overhead and compile recompiles at new shapes could eat the saving.
- W4: How much accuracy does low-res early training cost at 9 epochs? A loss of 0.5 pp or more makes it infeasible.

## Votes on open hypotheses
- H1: W1 supported, W2 supported, W3 supported, W4 supported
- H1.1: W1 supported, W2 supported, W3 supported, W4 supported
- H1.1.1: W1 refuted, W2 supported, W3 refuted, W4 supported
- H1.1.2: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H1.2: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H1.2.1: W1 unknown, W2 unknown, W3 refuted, W4 refuted
- H1.3: W1 supported, W2 supported, W3 supported, W4 supported
- H1.3.1: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H1.3.2: W1 supported, W2 supported, W3 supported, W4 supported
- H1.4: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H1.4.1: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H2: W1 supported, W2 supported, W3 supported, W4 unknown
- H2.1: W1 supported, W2 supported, W3 unknown, W4 supported
- H2.1.1: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H2.2: W1 supported, W2 supported, W3 supported, W4 unknown
- H2.2.1: W1 supported, W2 supported, W3 supported, W4 refuted
- H2.2.2: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H2.3: W1 refuted, W2 refuted, W3 refuted, W4 refuted
- H2.3.1: W1 refuted, W2 refuted, W3 refuted, W4 refuted
- H2.4: W1 refuted, W2 unknown, W3 unknown, W4 refuted
- H2.4.1: W1 refuted, W2 unknown, W3 refuted, W4 refuted
- H3: W1 unknown, W2 refuted, W3 refuted, W4 unknown
- H3.1: W1 supported, W2 supported, W3 supported, W4 unknown
- H3.1.1: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H3.2: W1 unknown, W2 unknown, W3 refuted, W4 unknown
- H3.2.1: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H4: W1 supported, W2 supported, W3 supported, W4 supported
- H4.1: W1 supported, W2 unknown, W3 unknown, W4 unknown
- H4.1.1: W1 unknown, W2 unknown, W3 unknown, W4 unknown
- H4.2: W1 supported, W2 supported, W3 unknown, W4 unknown
- H4.2.1: W1 unknown, W2 supported, W3 unknown, W4 unknown
