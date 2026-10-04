# Changes to karpathy/autoresearch

The research task is Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). We used upstream
commit `228791f`. These files are what the harness adds on top of it. `docs/SETUP.md` says how to apply them.

| file | what it is | origin |
|---|---|---|
| `sm120_attention.patch` | Lets `train.py` run on GPUs without FA3 kernels, such as sm_120: FlexAttention for the sliding-window layers, SDPA for the full-causal layers. Hopper keeps FA3. Also adds a one-line GPU note to `program.md`. | Our diff against upstream. Its context and `-` lines are upstream code, and it changes nothing else. |
| `harness_program.md` | The `program.md` the agent read in the RL run (R1) and the frz ablation (R3): one experiment per session, `./run.sh "<desc>"`, and the harness makes the keep/discard decision. | Adapted from upstream's `program.md`. The setup, rules and loop are rewritten for the harness. The goal, VRAM and simplicity-criterion paragraphs are upstream's wording. |
| `../tools/h2h/program.md` | The `program.md` for the continuous h2h sessions (R2), built into `canon.git` branch `h2h/start` by `tools/h2h/setup.py --create-start`. | Adapted from upstream's `program.md`. Most of its text is upstream's, and the run loop is changed to use `./run.sh` on a dedicated GPU. |

The two `program.md` files are prompts the agent reads, and they are byte-identical to the files used in our runs.
For that reason the attribution is here and not inside them: `tools/h2h/setup.py` checks that `h2h/start` holds
exactly `tools/h2h/program.md`. The upstream repository had no license file at the commits we used, and all credit
for the original text goes to its author.

Byte-identity with our runs: applying `sm120_attention.patch` to `228791f` gives the same `train.py` and `program.md`
blobs as our baseline commit (`c7666de` in our `canon.git`). `harness_program.md` is blob `bfaaee5` on that
repository's branches `autoresearch/oct3` and `frz/start`.
