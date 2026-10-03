m5-systems-paired: control 06022a1469ee, host-blocks ['h0-b0', 'h0-b1', 'h1-b0', 'h1-b1']
| config_id | params | n | time_s | d_time_% | se_% | acc |
|---|---|---|---|---|---|---|
| 06022a1469ee | {} | 20 | 5.350 | +0.00 | 0.00 | 0.7516 |
| 03b4868df065 | {"skip_discarded": true} | 20 | 5.413 | +1.19 | 0.12 | 0.7524 |
| 5886e1ac0784 | {"whiten_grad_off": true} | 20 | 5.188 | -3.02 | 0.25 | 0.7515 |
| 4b4ec89533c0 | {"compile_loss": true} | 20 | 5.291 | -1.10 | 0.29 | 0.7523 |
| d1b154b692ac | {"coordinate_descent": true} | 20 | 5.307 | -0.81 | 0.19 | 0.7512 |
| 543192064f53 | {"cudnn_benchmark_limit": 0} | 20 | 5.305 | -0.85 | 0.27 | 0.7517 |
| 23fde51185df | {"skip_discarded": true, "whiten_grad_off": true, "compile_loss": true, "coordinate_descent": true, "cudnn_benchmark_limit": 0} | 20 | 5.170 | -3.37 | 0.11 | 0.7522 |

per host-block time change vs control (%):
  h0-b0 [NVIDIA A100-SXM4-80GB, 500.00 W]: +1.29 -3.46 -1.56 -0.80 -1.28 -3.51
  h0-b1 [NVIDIA A100-SXM4-80GB, 500.00 W]: +1.47 -2.47 -0.81 -0.52 -0.58 -3.31
  h1-b0 [NVIDIA A100-SXM4-80GB, 400.00 W]: +0.90 -3.41 -1.58 -1.35 -1.32 -3.58
  h1-b1 [NVIDIA A100-SXM4-80GB, 400.00 W]: +1.11 -2.73 -0.43 -0.54 -0.21 -3.09
