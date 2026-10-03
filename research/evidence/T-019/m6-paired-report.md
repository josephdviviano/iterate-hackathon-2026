m6-exact-stack-paired: control e5cae98af74a, host-blocks ['h0-b0', 'h0-b1', 'h1-b0', 'h1-b1', 'h2-b0', 'h2-b1', 'h3-b0', 'h3-b1']
| config_id | params | n | time_s | d_time_% | se_% | acc |
|---|---|---|---|---|---|---|
| e5cae98af74a | {} | 40 | 5.744 | +0.00 | 0.00 | 0.7526 |
| 6bd99a471bea | {"whiten_grad_off": true} | 40 | 5.568 | -3.08 | 0.10 | 0.7530 |
| 6749dd3b7bbf | {"whiten_grad_off": true, "compile_loss": true} | 40 | 5.546 | -3.46 | 0.23 | 0.7520 |
| f3471d0a5240 | {"whiten_grad_off": true, "compile_loss": true, "coordinate_descent": true, "cudnn_benchmark_limit": 0} | 40 | 5.520 | -3.94 | 0.33 | 0.7523 |

per host-block time change vs control (%):
  h0-b0 [NVIDIA A100 80GB PCIe, 300.00 W]: -2.70 -2.74 -2.84
  h0-b1 [NVIDIA A100 80GB PCIe, 300.00 W]: -3.05 -4.18 -4.51
  h1-b0 [NVIDIA A100-SXM4-80GB, 500.00 W]: -3.57 -4.36 -4.86
  h1-b1 [NVIDIA A100-SXM4-80GB, 500.00 W]: -3.27 -3.68 -5.24
  h2-b0 [NVIDIA A100 80GB PCIe, 300.00 W]: -3.02 -2.85 -2.95
  h2-b1 [NVIDIA A100 80GB PCIe, 300.00 W]: -3.09 -2.81 -3.35
  h3-b0 [NVIDIA A100-SXM4-80GB, 400.00 W]: -3.14 -3.80 -4.50
  h3-b1 [NVIDIA A100-SXM4-80GB, 400.00 W]: -2.79 -3.29 -3.28
