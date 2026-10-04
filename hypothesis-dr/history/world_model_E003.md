# World model

_Last updated after: E003_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E001 @ 866d2a2: airbench96-style (whitening 2x2 stem, groups 128/384/576 depth 3 w/ residual, GELU, BN
weight frozen, bias lr x64), Nesterov SGD lr 9 / wd 0.012 / mom 0.85 / bs 1024, LS 0.3, lookahead, alternating
flip + translate 2, bf16 autocast + channels_last + torch.compile (warmed up in build), 10 epochs.
E001 (10 epochs): accuracy 0.7584, time 9.28 s.
E002 @ 479b9a3 = same at 9 epochs: accuracy 0.7552, time 8.34 s. FEASIBLE, current best (margin 0.2 pp).
Reference point from README: ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe (2 trials).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- time = mean(prepare+train) over 3 fixed seeds; accuracy = mean over the same 3 seeds. Fixed per-trial overhead
  of a trivial recipe is ~0.1 s (E000), so the score is essentially all training compute.
- Feasibility gate is 0.753 on 3 seeds (official: 0.75 on 40). README L40 calibration: per-trial accuracy std
  ~0.25 pp, so a 3-seed mean has std ~0.15 pp; 0.753 is a ~2-sigma margin. Time std ~0.14 s at ~60 s.
- Dev GPU is A100-SXM4 (official: A100 PCIe, lower power cap -> expect official times somewhat slower).
- No TTA allowed: eval is a single view through model.forward in eval mode.

## Where the cost goes
<!-- what dominates the objective, and why -->
- (E001) prepare is 0.06-0.15 s (<2%); train is ~9.16 s for 480 steps = 19 ms/step at bs 1024.
  ~0.6 GFLOP fwd/img -> ~95 TFLOP/s achieved, ~30% of A100 bf16 peak. Time is ~linear in epochs x net FLOPs.
- Per-trial time std is tiny (9.24-9.30 s across seeds).
- (E003 profile) The step is GPU-bound, not launch-bound: 18.8 ms/step on real batches; per step conv bwd ~8.5 ms,
  conv fwd ~4.1 ms (cuDNN, ~75%), Triton elementwise (BN/GELU/pool) ~3.4 ms, adaptive_max_pool2d bwd 0.48 ms,
  SGD 0.44 ms. Per-run overheads (augment, permute, lookahead) ~0.16 s (~2%). Prepare ~0.06-0.16 s.
  => Time can only fall substantially by fewer steps (epochs), fewer FLOPs per step (narrower/shallower net,
  lower resolution), or faster convs. Optimizer/launch fusion is worth <=3%.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- The airbench96-style base without TTA clears 0.753 at 10 epochs with ~0.5 pp margin (E001).
- Per-seed accuracy spread at 10 epochs: 0.7568-0.7613 (range 0.45 pp) (E001); at 9 epochs 0.7546-0.7558 (E002).
- Run-to-run noise with fixed seeds: accuracy mean bit-identical, time +-1% (E002 vs E003). Differences
  > ~0.1 pp / > ~2% are real effects of the change (on these 3 seeds).
- Time is linear in epochs: ~0.93 s/epoch + ~0.1 s fixed (E001, E002).
- Accuracy vs epochs near the knee: ~0.3 pp/epoch between 9 and 10 epochs (E001, E002), flatter than the
  0.4-0.6 pp/epoch the tree assumed. So 1 epoch (~10% time) buys only ~0.3 pp: any change that costs <10% time
  per 0.3 pp gained is worth it, and any change that loses >0.3 pp must save >1 epoch.

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): acc 0.012, time 0.11 s -> harness works; per-trial overhead ~0.1 s; need a real recipe.
E001 (I001, H1.1.1): acc 0.7584, time 9.28 s, feasible -> base recipe established; prepare negligible; ~30% MFU.
E002 (I003, H1.1.3): 9 epochs acc 0.7552, time 8.34 s, feasible (inconclusive: 0.31 pp/epoch) -> slope flatter than assumed; knee ~8.3-8.8 epochs.
E003 (I004, H4.1.1): rerun of E002: acc identical, time +0.9% -> noise floor tiny; profile shows GPU-bound convs (~97% of time).
