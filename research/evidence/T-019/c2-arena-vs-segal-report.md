c2-arena-vs-segal: reference segal; host GPUs ['NVIDIA A100-SXM4-80GB']
| arm | trials | acc % (SE) | time s | d_time % vs ref (SE) | d_time % PCIe only |
|---|---|---|---|---|---|
| segal | 40 | 75.35 (0.04) | 5.132 | +0.00 (0.00) | +nan (nan) n=0 |
| arena-I001 | 40 | 75.27 (0.04) | 7.283 | +41.90 (0.22) | +nan (nan) n=0 |
| arena-I044 | 20 | 75.66 (0.06) | 5.500 | +8.43 (0.31) | +nan (nan) n=0 |
| arena-I049 | 30 | 75.40 (0.05) | 5.257 | +3.10 (0.41) | +nan (nan) n=0 |

Build failures (harness build_timeout = 600 s): h0 arena-I044, h2 arena-I044, h3 arena-I049; all other 13 runs complete.
