# Idea ledger

Success 12 (confirmed on fresh seeds with paired timing: 4), failed 38, vetoed 1, pending 35. Audited rejections 7, of which false rejects 0. Screening rows decide on accuracy only (unpaired time); a success counts for adoption once confirmed.

| Round | Id | Levers | Committee p / pred dpp / pred dtime | Decision | Measured dpp / dtime (n) | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | r1-L1 | `{"switch_widths": [128, 384, 512]}` | 0.07 / -0.30 / -0.035 | audit | -0.36 / -0.051 (10) | failed |
| 1 | r1-L2 | `{"switch_widths": [128, 320, 640]}` | 0.05 / -0.28 / -0.030 | audit | -0.25 / -0.053 (10) | failed |
| 1 | r1-L3 | `{"stage1_cooldown": [0.5, 0.8]}` | 0.04 / -0.26 / -0.001 | audit | -0.13 / +0.145 (10) | failed |
| 1 | r1-L4 | `{"bias_scaler_final": 16.0}` | 0.10 / +0.04 / +0.000 | run | +0.06 / +0.025 (10) | failed |
| 1 | r1-L5 | `{"bias_scaler": 128.0, "bias_scaler_final": 16.0}` | 0.06 / +0.01 / +0.000 | run | -0.27 / +0.004 (10) | failed |
| 1 | r1-L6 | `{"lr_shape": "wsd_sqrt", "lr_decay_start": 0.6}` | 0.07 / -0.00 / +0.000 | run | -0.12 / +0.001 (10) | failed |
| 1 | r1-L7 | `{"stage_depths": [3, 2, 3]}` | 0.07 / -0.53 / -0.083 | audit | -0.30 / -0.106 (10) | failed |
| 1 | r1-L8 | `{"stage_depths": [3, 3, 2]}` | 0.07 / -0.39 / -0.043 | audit | -0.92 / -0.065 (10) | failed |
| 1 | r1-L9 | `{"init_gain": 0.5, "bias_scaler": 32.0}` | 0.12 / +0.05 / +0.000 | run |  | pending |
| 1 | r1-L10 | `{"switch_momentum_scale": 0.25}` | 0.04 / +0.00 / +0.000 | run |  | pending |
| 1 | r1-L11 | `{"bn_momentum": 0.4}` | 0.04 / +0.00 / +0.000 | run | -0.12 / +0.045 (10) | failed |
| 1 | r1-L12 | `{"widths": [112, 384, 640]}` | 0.11 / -0.17 / -0.034 | run | -0.03 / +0.104 (10) | failed |
| 1 | r1-L13 | `{"widths": [128, 384, 576], "epochs": 8.5}` | 0.04 / +0.02 / +0.001 | run | -0.14 / +0.026 (10) | failed |
| 1 | r1-L14 | `{"lookahead_power": 4.0}` | 0.06 / +0.00 / +0.000 | carry |  | pending |
| 1 | r1-L15 | `{"weight_decay": 0.0199}` | 0.07 / -0.01 / +0.000 | carry |  | pending |
| 1 | r1-L16 | `{"label_smoothing": 0.3, "momentum": 0.8}` | 0.08 / +0.03 / +0.000 | carry |  | pending |
| 2 | r2-L1 | `{"stage1_cooldown": [0.4, 0.7]}` | 0.04 / -0.26 / -0.003 | audit | -0.35 / +0.004 (10) | failed |
| 2 | r2-L2 | `{"stage1_cooldown": [0.55, 0.85]}` | 0.05 / -0.03 / +0.000 | run | -0.07 / +0.005 (10) | failed |
| 2 | r2-L3 | `{"bias_scaler_final": 32.0}` | 0.12 / +0.04 / +0.000 | run | -0.04 / +0.003 (10) | failed |
| 2 | r2-L4 | `{"bias_scaler": 96.0, "bias_scaler_final": 24.0}` | 0.07 / +0.00 / +0.000 | run | -0.12 / +0.003 (10) | failed |
| 2 | r2-L6 | `{"label_smoothing": 0.2, "ls_end": 0.35}` | 0.08 / +0.02 / +0.000 | run | +0.11 / +0.003 (10) | failed |
| 2 | r2-L10 | `{"switch_momentum_scale": 0.7}` | 0.05 / +0.03 / +0.000 | run | -0.02 / +0.003 (10) | failed |
| 2 | r2-L12 | `{"res_schedule": [[0.0, 20], [0.35, 24], [0.55, 32]]}` | 0.09 / +0.02 / +0.015 | run | -0.19 / -0.001 (10) | failed |
| 2 | r2-L13 | `{"freeze_schedule": [[0.88, 1]]}` | 0.07 / -0.12 / -0.015 | run | -0.14 / -0.035 (10) | failed |
| 2 | r2-L14 | `{"lookahead_every": 8}` | 0.04 / -0.01 / +0.000 | run | -0.11 / +0.002 (10) | failed |
| 2 | r2-L15 | `{"translate": 3}` | 0.06 / -0.07 / +0.000 | run | -0.30 / +0.004 (10) | failed |
| 2 | r2-L16 | `{"pskd_alpha": 0.15}` | 0.04 / +0.00 / +0.005 | run | -0.14 / +0.025 (10) | failed |
| 3 | r3-L1 | `{"label_smoothing": 0.2, "ls_end": 0.45}` | 0.10 / +0.01 / +0.003 | run | +0.05 / -0.011 (10) | failed |
| 3 | r3-L2 | `{"label_smoothing": 0.15, "ls_end": 0.35}` | 0.06 / -0.01 / +0.003 | run | +0.12 / -0.023 (10) | failed |
| 3 | r3-L3 | `{"ls_end": 0.35, "bias_scaler_final": 16.0}` | 0.08 / +0.04 / +0.004 | run | +0.10 / +0.070 (10) | failed |
| 3 | r3-L4 | `{"bias_scaler_final": 8.0}` | 0.06 / +0.01 / +0.000 | run | +0.19 / +0.064 (10) | success |
| 3 | r3-L5 | `{"bias_scaler": 48.0}` | 0.10 / +0.04 / +0.000 | run | +0.14 / +0.089 (10) | failed |
| 3 | r3-L6 | `{"bias_scaler": 32.0, "ls_end": 0.35}` | 0.09 / +0.04 / +0.002 | run | +0.34 / -0.021 (10) | success |
| 3 | r3-L7 | `{"whiten_bias_epochs": 10.0}` | 0.04 / +0.01 / +0.009 | run | +0.07 / +0.025 (10) | failed |
| 3 | r3-L8 | `{"head_wd_mult": 0.0}` | 0.05 / +0.00 / +0.000 | run | +0.07 / +0.073 (10) | failed |
| 3 | r3-L9 | `{"init_gain": 0.4}` | 0.06 / +0.03 / +0.000 | run | -0.02 / -0.025 (10) | failed |
| 3 | r3-L10 | `{"init_gain": 0.5, "ls_end": 0.35}` | 0.07 / +0.03 / +0.003 | run | +0.18 / -0.013 (10) | success |
| 3 | r3-L11 | `{"res_blend_steps": 12}` | 0.04 / +0.03 / +0.004 | run | +0.11 / +0.001 (10) | failed |
| 3 | r3-L12 | `{"res_schedule": [[0.0, 22], [0.5, 32]]}` | 0.06 / +0.04 / +0.030 | carry |  | pending |
| 3 | r3-L13 | `{"res_schedule": [[0.0, 18], [0.45, 32]]}` | 0.07 / -0.04 / -0.006 | carry |  | pending |
| 3 | r3-L14 | `{"epochs": 8.0, "ls_end": 0.35}` | 0.09 / -0.14 / -0.030 | carry |  | pending |
| 3 | r3-L15 | `{"lr": 12.5, "momentum": 0.8}` | 0.04 / +0.01 / +0.000 | carry |  | pending |
| 3 | r3-L16 | `{"lr_peak_frac": 0.17, "lr_end": 0.04}` | 0.06 / +0.01 / +0.000 | carry |  | pending |
| 3 | r3-L6c | `{"label_smoothing": 0.2, "ls_end": 0.35}` | 0.08 / +0.02 / +0.000 | confirm | -0.04 / +0.004 (10) | failed |
| 4 | r4-L4c | `{"bias_scaler_final": 8.0}` | 0.06 / +0.01 / +0.000 | confirm | +0.12 / +0.003 (20) | failed |
| 4 | r4-L6c | `{"bias_scaler": 32.0, "ls_end": 0.35}` | 0.09 / +0.04 / +0.002 | confirm | +0.19 / +0.003 (20) | success |
| 4 | r4-L10c | `{"init_gain": 0.5, "ls_end": 0.35}` | 0.07 / +0.03 / +0.003 | confirm | +0.01 / +0.004 (20) | failed |
| 4 | r4-stack | `{"bias_scaler_final": 8.0, "bias_scaler": 32.0, "ls_end": 0.35, "init_gain": 0.5}` |  | confirm | +0.17 / +0.002 (20) | success |
| 5 | r5-L1 | `{"bias_scaler": 24.0, "ls_end": 0.35}` | 0.10 / +0.03 / +0.000 | run | +0.34 / +0.002 (10) | success |
| 5 | r5-L2 | `{"bias_scaler_final": 4.0}` | 0.09 / +0.03 / +0.000 | run | +0.31 / +0.010 (10) | success |
| 5 | r5-L3 | `{"bias_scaler": 32.0, "bias_scaler_final": 8.0}` | 0.09 / +0.04 / +0.001 | run | +0.39 / -0.016 (10) | success |
| 5 | r5-L4 | `{"bias_scaler_final": 8.0, "ls_end": 0.35}` | 0.19 / +0.09 / +0.000 | run | +0.15 / +0.009 (10) | success |
| 5 | r5-L5 | `{"label_smoothing": 0.1, "ls_end": 0.4}` | 0.06 / -0.01 / +0.000 | run | +0.03 / -0.002 (10) | failed |
| 5 | r5-L6 | `{"label_smoothing": 0.15, "ls_end": 0.5}` | 0.05 / -0.02 / +0.000 | carry |  | pending |
| 5 | r5-L7 | `{"res_blend_steps": 24}` | 0.07 / +0.03 / +0.002 | run | +0.16 / +0.083 (10) | success |
| 5 | r5-L8 | `{"res_blend_steps": 12, "ls_end": 0.35}` | 0.04 / -0.02 / +0.003 | run | +0.09 / +0.010 (10) | failed |
| 5 | r5-L9 | `{"res_blend_steps": 12, "bias_scaler": 32.0}` | 0.07 / +0.04 / +0.001 | carry |  | pending |
| 5 | r5-L10 | `{"init_gain": 0.5, "bias_scaler_final": 8.0}` | 0.08 / +0.05 / +0.000 | carry |  | pending |
| 5 | r5-L11 | `{"head_wd_mult": 0.0, "ls_end": 0.35}` | 0.06 / -0.05 / +0.000 | carry |  | pending |
| 5 | r5-L12 | `{"momentum": 0.8, "bias_scaler": 32.0}` | 0.07 / +0.04 / +0.000 | carry |  | pending |
| 5 | r5-L13 | `{"whiten_bias_epochs": 10.0, "bias_scaler_final": 8.0}` | 0.07 / +0.01 / +0.001 | carry |  | pending |
| 5 | r5-L14 | `{"epochs": 7.75, "bias_scaler": 32.0}` | 0.05 / -0.30 / -0.060 | rejected |  | vetoed |
| 5 | r5-L15 | `{"epochs": 8.0, "bias_scaler_final": 8.0}` | 0.16 / -0.09 / -0.030 | carry |  | pending |
| 5 | r5-L16 | `{"stage_depths": [2, 3, 3], "bias_scaler": 32.0}` | 0.17 / -0.22 / -0.055 | audit | -0.09 / -0.076 (10) | failed |
| 5 | r5-L2c | `{"label_smoothing": 0.15, "ls_end": 0.35}` | 0.06 / -0.01 / +0.003 | confirm | +0.12 / +0.008 (10) | failed |
| 5 | r5-L3c | `{"ls_end": 0.35, "bias_scaler_final": 16.0}` | 0.08 / +0.04 / +0.004 | confirm | +0.19 / +0.030 (10) | success |
| 5 | r5-L5c | `{"bias_scaler": 48.0}` | 0.10 / +0.04 / +0.000 | confirm | +0.19 / +0.019 (10) | success |
| 5 | r5-L11c | `{"res_blend_steps": 12}` | 0.04 / +0.03 / +0.004 | confirm | +0.02 / +0.006 (10) | failed |
| 6 | r6-ep8.25 | `{"bias_scaler": 32.0, "ls_end": 0.35, "epochs": 8.25}` |  | confirm |  | pending |
| 6 | r6-ep8.1 | `{"bias_scaler": 32.0, "ls_end": 0.35, "epochs": 8.1}` |  | confirm |  | pending |
| 6 | r6-ep8.0 | `{"bias_scaler": 32.0, "ls_end": 0.35, "epochs": 8.0}` |  | confirm |  | pending |
| 6 | r6-ep7.9 | `{"bias_scaler": 32.0, "ls_end": 0.35, "epochs": 7.9}` |  | confirm |  | pending |
| 7 | r7-L1 | `{"bias_scaler": 32.0, "bias_scaler_final": 4.0}` | 0.31 / +0.14 / +0.000 | run |  | pending |
| 7 | r7-L2 | `{"bias_scaler": 24.0, "bias_scaler_final": 8.0}` | 0.09 / +0.04 / +0.000 | run |  | pending |
| 7 | r7-L3 | `{"bias_scaler_final": 2.0}` | 0.28 / +0.12 / +0.000 | run |  | pending |
| 7 | r7-L4 | `{"epochs": 8.0, "bias_scaler_final": 4.0}` | 0.34 / -0.04 / -0.029 | run |  | pending |
| 7 | r7-L5 | `{"epochs": 8.0, "bias_scaler": 24.0}` | 0.25 / -0.07 / -0.030 | run |  | pending |
| 7 | r7-L6 | `{"epochs": 8.5, "stage_depths": [2, 3, 3]}` | 0.18 / -0.07 / -0.028 | run |  | pending |
| 7 | r7-L7 | `{"whiten_grad_off": true}` | 0.82 / +0.00 / -0.030 | run |  | invalid |
| 7 | r7-L8 | `{"compile_loss": true, "coordinate_descent": true}` | 0.21 / +0.00 / -0.018 | run |  | invalid |
| 7 | r7-L9 | `{"cudnn_benchmark_limit": 0, "whiten_grad_off": true}` | 0.59 / +0.00 / -0.031 | run |  | pending |
| 7 | r7-L10 | `{"label_smoothing": 0.25, "bias_scaler": 32.0}` | 0.08 / +0.04 / +0.000 | run |  | pending |
| 7 | r7-L11 | `{"momentum": 0.8, "ls_end": 0.35}` | 0.04 / -0.04 / +0.000 | run |  | pending |
| 7 | r7-L12 | `{"head_wd_mult": 0.0, "bias_scaler": 32.0}` | 0.09 / +0.03 / +0.000 | run |  | pending |
| 7 | r7-L13 | `{"whiten_bias_epochs": 10.0, "bias_scaler": 32.0}` | 0.09 / +0.04 / +0.007 | carry |  | pending |
| 7 | r7-L14 | `{"res_blend_steps": 16, "bias_scaler": 32.0}` | 0.08 / +0.04 / +0.006 | carry |  | pending |
| 7 | r7-L15 | `{"scaling_factor": 0.14, "ls_end": 0.35}` | 0.04 / -0.11 / +0.000 | carry |  | pending |
| 7 | r7-L16 | `{"init_gain": 0.4, "bias_scaler": 32.0}` | 0.09 / +0.04 / +0.000 | carry |  | pending |
