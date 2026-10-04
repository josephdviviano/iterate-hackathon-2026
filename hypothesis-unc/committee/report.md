# Uncertainty committee

Calibration: 61 results scored; consensus range hit 72% (target ~80%); experimenter's sealed range hit 92%; counterexamples 17; disagreement predicts misses: AUROC 0.47 (not informative: priorities ignore it); improve votes pick keeps: AUROC 0.74; recent bias -0.26 half-ranges; veto active (audits: [('E045', 'discard')]).

## Members
- W1 retired (mechanism, claude-opus-5-5, since E000): hit 61% of 38, recent 50%; retired: W1 hit 25% of its last 8 forecasts
- W2 active (empirical, claude-sonnet-5-5, since E000): hit 72% of 61, recent 75%
- W3 active (cost, claude-opus-5-5, since E000): hit 74% of 61, recent 62%
- W4 retired (skeptic, claude-sonnet-5-5, since E000): hit 62% of 16, recent 75%; retired: W4 hit 25% of its last 8 forecasts
- W5 active (contrarian, claude-opus-5-5, since E011): hit 74% of 50, recent 50%
- W6 active (counterexample, claude-sonnet-5-5, since E032): hit 67% of 30, recent 50%

## Queued ideas
idea | median range | conflict | improve votes | harm votes | verdict entropy
---|---|---|---|---|---
I097 | [4.9350000000000005, 5.005, 5.1] | 0.0 | 2/4 | 0/4 | 0.631
I105 | [5.04, 5.11, 5.1899999999999995] | 0.0 | 0/4 | 4/4 | 0.0
I106 | [4.96, 5.025, 5.1] | 0.0 | 2/4 | 0/4 | 0.0
I107 | [4.9, 4.98, 5.0649999999999995] | 0.0 | 3/4 | 0/4 | 0.512
I108 | [4.96, 5.025, 5.1] | 0.0 | 0/4 | 0/4 | 0.512
I109 | [4.96, 5.03, 5.1] | 0.0 | 1/4 | 0/4 | 0.512

## Counterexamples
- E031 (I044): Explore: proposes a new H2.4 child. Drop easy examples using free per-sample losses from the main net, with no proxy. The committee vetoed I041 for the proxy's cost; this version adds no extra forward -> measured time=5.249974399999094, outside every member's range
- E032 (I049): Exploit, a time-neutral booster that targets the end-point, where averaging forms (E005 showed the late phase is worth about 1 pp). On the E027 tip, keep label smoothing at 0.3 for the first 85% of st -> measured accuracy=0.7483, outside every member's range
- E043 (I058): Explore diagnostic, with no recipe change. It sets the value of the remaining stem levers (H2.6.4/H2.6.5/H2.6.3) and the PCIe-transfer caveat in H4.3. Offline, run torch.profiler on the tip's compiled -> measured accuracy=0.7496666666666667, outside every member's range
- E046 (I070): Exploit, staggered FreezeOut cascade. This is the only remaining backward-truncation lever that targets the largest block, group 2 (~51% of conv MACs). On the E036 tip, put the group-2 conv filters (M -> measured time=5.0786617846655036, outside every member's range
- E056 (I086): Sharpened replacement for I078, gated on the committee's shared crux (do ReLU's saved bytes reach 3%?). Step 1 is offline, with no repo change. On the E049 tip code with synthetic channels_last batch- -> measured time=4.952750553997855, outside every member's range
- E062 (I096): Explore, the largest untested optimizer axis. It is now cheap to judge because the break-even loss is about 0.25 pp at the 0.77 s/pp dial. On the E049 tip, replace Muon on every conv filter group (gro -> measured time=4.743637117329247, accuracy=0.7237, outside every member's range

## Cruxes (latest round)
- W2: Does a time-neutral Muon lr or momentum retune give at least +0.1 pp on 60 draws? That would be the only remaining free margin.
- W2: What is the true population mean of the tip on fresh seeds? If it is near 0.7525, the 0.753 3-seed bar leaves no room for further cuts.
- W2: Is the post-225-step dial about 0.9 s/pp, as E064 suggests (−0.185 s for about −0.2 pp), or steeper?
- W3: Was E064's −0.22 pp at 213 steps real (are the free early steps exhausted), or 3-seed noise? A 40-draw re-read decides whether any further step trimming is possible.
- W3: Does Muon lr ×1.5 give a real ≥0.12 pp gain on 60 draws? It is the only plausible time-neutral booster that could fund another ~0.1–0.15 s cut.
- W3: How much of the ~0.1 s non-NS optimizer overhead can a fused, batched Muon path recover? That is the last free time lever with identical math.
- W5: Is a 20 px step host- and launch-bound? Compare the wall time per step with the GPU kernel time per step over a whole real 20 px epoch. If the gap is ≥5 ms per step, I107 and larger low-res batches (H5.1.3) are the best time levers.
- W5: Does merging the Muon and SGD calls with GPU-resident lr tables (I107) cut ≥0.05 s, or does E047's 5% idle bound it?
- W5: Is the clean train top-1 of the averaged weights below 92% (under-fitting)? That decides whether boosters should come from the optimizer or from regularization.
- W6: Is the early-trim cost real? E064 read -0.2 pp at 213 steps, so what does a 40-draw population measurement give?
- W6: Are Muon lr and momentum off-optimum at about 225 steps? This is the only near-free booster left.
- W6: Is the recipe under-fit (train top-1 at most 90%) or saturated? This decides whether regularizer or fitting-speed changes are worth testing.

## Votes on open hypotheses
- H1.3.5: W2 unknown, W3 refuted, W5 refuted, W6 unknown
- H1.5: W2 unknown, W3 supported, W5 supported, W6 unknown
- H1.5.4: W2 unknown, W3 supported, W5 supported, W6 supported
- H2.3.4: W2 refuted, W3 unknown, W5 refuted, W6 refuted
- H4.3: W2 supported, W3 supported, W5 supported, W6 supported
- H5: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H5.1: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H5.1.1: W2 unknown, W3 unknown, W5 unknown, W6 refuted
- H5.1.2: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H1.5.5: W2 refuted, W3 unknown, W5 refuted, W6 refuted
- H1.5.6: W2 refuted, W3 refuted, W5 refuted, W6 unknown
- H1.5.7: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H2.3.5: W2 unknown, W3 unknown, W5 unknown, W6 supported
- H6.2: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H6.2.1: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H2.3.7: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H1.5.8: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H5.1.3: W2 unknown, W3 refuted, W5 unknown, W6 refuted
- H2.6.6: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H5.2: W2 supported, W3 supported, W5 supported, W6 supported
- H5.2.1: W2 unknown, W3 supported, W5 unknown, W6 supported
- H4.1.6: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H6.5: W2 supported, W3 unknown, W5 supported, W6 supported
- H6.6: W2 supported, W3 supported, W5 supported, W6 supported
- H6.6.1: W2 supported, W3 supported, W5 unknown, W6 supported
- H1.7: W2 supported, W3 supported, W5 supported, W6 supported
- H1.5.9: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H7: W2 supported, W3 supported, W5 supported, W6 supported
- H7.1: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H7.1.1: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H2.6.10: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H6.5.2: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H6.5.3: W2 refuted, W3 refuted, W5 unknown, W6 refuted
- H1.8: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H1.8.1: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H1.5.10: W2 unknown, W3 supported, W5 refuted, W6 unknown
- H8: W2 supported, W3 supported, W5 supported, W6 supported
- H8.1: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H8.1.1: W2 refuted, W3 refuted, W5 refuted, W6 refuted
- H8.2: W2 supported, W3 unknown, W5 supported, W6 supported
- H8.2.1: W2 supported, W3 unknown, W5 supported, W6 supported
- H1.2.4: W2 unknown, W3 unknown, W5 unknown, W6 unknown
- H2.2.5: W2 unknown, W3 unknown, W5 supported, W6 refuted
- H2.6.12: W2 unknown, W3 refuted, W5 refuted, W6 refuted
- H2.6.13: W2 unknown, W3 supported, W5 unknown, W6 supported
- H4.7: W2 supported, W3 supported, W5 supported, W6 supported
- H4.7.1: W2 supported, W3 supported, W5 supported, W6 supported
- H4.2.7: W2 supported, W3 supported, W5 unknown, W6 supported
- H4.2.8: W2 refuted, W3 refuted, W5 refuted, W6 supported
- H6.5.4: W2 supported, W3 refuted, W5 refuted, W6 supported
