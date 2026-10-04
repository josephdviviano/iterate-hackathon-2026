<!-- Frozen red-team review (round 6); model output, not user input. -->

I found no rule violations and no evidence that the entry fails to qualify. The main gap is that nothing adopted since M8 has run on the pinned PyTorch 2.4 / A100 stack. That covers the stage-1 freeze, depth-2 stage 2, label smoothing 0.4 at 8.25 epochs, the timed-path fixes and the full-size warm-up. The 4.7–4.8 s time, the build time and the absence of recompiles are all extrapolated from the local Blackwell / torch 2.7.1 machine. I modified no files and ran no GPU jobs; the only check of the machine was a read-only `nvidia-smi` query.

## Ranked issues

**1. HIGH: the entry's code has never run on PyTorch 2.4, and the frozen path depends on the torch version.**
- **Evidence:**
  - The newest A100 runs are M8a and M8b, which predate D-012. No Modal sweep since then has used the freeze, depth or label-smoothing settings.
  - The frozen branch in `submissions/team_segal/model.py` slices the module list: `self.groups[:1](...)` and `self.groups[1:](x)`.
  - On 2.7.1 the compiler inlines built-in modules (`inline_inbuilt_nn_modules=True`); on 2.4.0 it does not (`False`). In 2.4 (`torch/_dynamo/variables/nn_module.py:672-697`), slicing builds a new `nn.Sequential` on every call and registers it under a slice source, so the guard behaviour differs from the dev stack.
  - If the guards fail on every call, the warm-up would recompile until it hits `cache_size_limit=8`. The frozen 20% of steps would then fall back to eager silently. That costs time, not accuracy.
- **Cheap check:** run a short CPU job in `.venv` (torch 2.4) with `torch.compile(backend="aot_eager")` and `TORCH_LOGS=recompiles`. Step it through the bias-gradient, resolution and freeze changes twice and count compiled graphs; 4 is the expected answer. This takes seconds.
- **Exact fix if it fails:** replace the slices with `self.groups[0](...)` and `self.groups[2](self.groups[1](x))`. The computation is identical.

**2. MEDIUM-HIGH: the cold build time under a hard 4-CPU limit is unmeasured for the current entry.**
- **Evidence:**
  - The official command is `docker run --cpus 4` (ORGANIZERS.md), which caps every process, including the compiler's worker processes.
  - Modal used `cpu=4.0` (`research/modal_a100.py`), which is a request under gVisor with 20 CPUs visible. I believe Modal treats this as a soft limit that can burst; that needs confirming.
  - M8 built in 117 s on the PCIe host and 195 s on the SXM host, with 3 graphs.
  - The current entry adds a 4th graph and a full 50k-image, 396-step warm-up.
  - Every local build took 4–9 s only because the compiler caches were already warm. For comparison, F-035 recorded builds up to 409 s.
- **Cheap check:** do a local cold build with fresh `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` under `systemd-run --scope -p CPUQuota=400%` (or `docker --cpus 4`).

**3. MEDIUM: the accuracy margin is thinner than "5 SE".**
- **Pooled results at the current defaults:**
  - 177 fresh seeds (S61, S62, the S63/S64/S65 controls, C3, and the first 17 S66 control trials while S66 is still running): **75.160%** (SD 0.234, SE 0.018).
  - Adding S58, which was the screen the setting was picked from: 217 seeds, 75.168%.
  - The blocks are consistent with each other (chi² about 6.4 on 9 degrees of freedom).
- **Why "5 SE" overstates it:** F-062 uses only the SE of the local estimate. It leaves out the official run's own 40-seed noise (about 0.038) and the uncertainty in the local-to-A100 offset.
- **The offset evidence is thinner than F-051 suggests:**
  - "M7a vs S41" compares A100 with A100: S41's matching cell `4fe84eaa1ccd` ran on a Modal A100-SXM4.
  - The real local-vs-A100 pairs are M3/S29 (−0.043 pp, SE 0.035), M8 vs local (+0.01) and M7 vs S43 control (+0.06). Pooled, that is about 0 ± 0.02.
- **Estimated P(official 40-seed mean < 75%)**, with a predictive SD of about 0.056 including 0.03 for unexplained stack effects:

| Scenario | P(fail) |
|---|---|
| Pooled offset of about 0 | about 0.2% |
| M3 offset (−0.043) applies | about 3% |
| No offset or stack uncertainty | about 0.01% |

- **Trend:** the margin has fallen each round: 75.29 (M7a), then 75.21 (M8-config, local and A100), then 75.16 now.

**4. MEDIUM: winner's curse shrank both recent adoptions.**
- **D-012:** chosen at 75.199% (S47, 6 arms). On 340 fresh seeds it averages 75.158%. The M8 config averages about 75.20–75.21% locally. So "at control accuracy" overstates it: the real cost is about −0.05 ± 0.025 pp, roughly 1% of time at your exchange rate. It is still clearly worth the roughly 10% saving.
- **D-013:** the S58 screen gave +0.08 pp over control, but the fresh S61 run gave +0.00. The 80-seed "+0.04" includes the screen it was selected from. Label smoothing 0.4 at 8.25 epochs is a sound 2.5% step cut at about zero accuracy cost. The claim that "accuracy still rises from 0.3 to 0.4" is within noise.
- Neither adoption is a false positive in direction.

**5. MEDIUM: whether the freeze/depth time saving carries over to the A100 PCIe.**
- **Local evidence is solid:**
  - S46 gives −13.7% for the stack at 8.25 epochs, with the two GPUs within 1.3 points.
  - S48 gives −9.2% / −12.2% at 8.5 epochs, where GPU identity is confounded with arm order.
  - Medians and dropping the first trial don't change these.
- **Transfer argument:** both changes remove whole layers or whole backward passes, and both cards run power-capped near 300 W. That differs from H1, crops and narrow widths, which changed shapes, so I expect the saving to carry over.
- **Estimate:** 5.526 × 0.893 × 0.983 × 0.975 ≈ **4.73 s**, about ±0.2 s at one SD once you allow for transfer and host variation.

**6. LOW-MEDIUM: local paired timing has bias sources.**
- **Thermal drift:** S62a rises from 4.028 to 4.266 s (+5.9%) over 20 trials. The GPUs sit at 85–88 °C and 1450–1620 MHz at the 300 W cap. Arms that run first get a "cool" advantage of about ±1.5%.
- **GPU 0 sharing:** GPU 0 hosts two `text-embeddings-router` processes (about 3.6 GB). It runs 60–80 MHz slower with noisier times (SD 0.078 vs 0.049 in S62; an S61 control trial jumped to 5.445 s).
- **Fix:** ABBA blocks of 5 trials, GPU 1 only (or stop the embedding services), and a soak period before timing.

**7. LOW: F-060's timing numbers are inflated by compile spikes.**
- **Evidence:** in three S60 arms the first trial took 11.6–12.2 s. These are the source of the reported +19.8% and +23.1% / +18.0%.
- **Robust deltas:** +1.3% (1×1 residual), +3.3% (stage-1 width 96) and +2.4% (both).
- **Effect on the decision:** set against S59's +0.04 / +0.12 / +0.18 pp, the combined arm is about break-even. Rejecting it still stands, but the stated mechanism ("narrow shapes are not cheaper") is partly an artifact, and the lab warm-up missed a graph for those arms.

**8. LOW: rule compliance (section 3).**
- **No violations:**
  - Build uses only synthetic data and seed 0.
  - The harness reseeds the global RNG before each prepare, and the per-trial generator is created fresh.
  - Parameters, buffers, optimiser and lookahead are all reset or rebuilt each trial, and the gradient and freeze flags are restored at the end of training.
  - The trimmed tail is fixed in advance.
  - Evaluation is a single eager forward pass in eval mode.
  - The DCT basis is analytic, not learned.
- **Bit-identity:** S57 is bit-identical per seed (10/10 on each GPU), which supports the organisers' reordered-seed check.
- **Housekeeping:** make sure the untracked `submissions/team_codex/` does not go into the PR.
- **Evaluation stability:** the zero-variance BN behaviour behind F-065 has not appeared at evaluation in any current-like trial.

**9. LOW: the README's fallbacks (8.5 / 8.75 epochs) were never measured with label smoothing 0.4.** One local 40-seed run at 8.5 epochs would put a measured fallback on file.

## Go / no-go

**Go, conditionally.** The entry very probably qualifies: about 0.2% chance of failing at the central estimate, up to about 3% if the M3 offset applies. It is probably about 14% faster than M8.

Minimum confirmation before submitting:
1. Run the CPU torch-2.4 compile check from item 1 (minutes).
2. Do a local cold build under a 400% CPU quota (item 2).
3. Run one official-equivalent **M9** on an A100 80GB PCIe: pinned container, hard 4-CPU limit, cold caches, 40 fresh seeds, recompile logging on, `--official` if the Modal environment passes its checks. This is the one essential Modal run, because the build time, recompiles, time and accuracy on PyTorch 2.4 cannot be measured locally.
4. Decide in advance to switch to the measured 8.5-epoch fallback if M9's accuracy comes in below about 75.10%.

The key file is `submissions/team_segal/model.py` (the `Net.forward` frozen branch). Raw per-trial data comes from `results/sweeps/{s58,s61,s62a,s62b,s63,s64,s65,s66,c3a-segal,c3b-segal,s46*,s48*,s60*,m3,m7*,m8*}`.
