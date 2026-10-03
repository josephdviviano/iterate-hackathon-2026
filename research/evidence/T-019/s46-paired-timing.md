# S46 local paired timing (GPU 0 order a, GPU 1 reversed order b; 10 seeds each)

| arm | GPU0 d% | GPU1 d% | mean d% |
|---|---|---|---|
| depth-2 stage 2, 8.25 ep | -6.7 | -7.0 | -6.9 |
| stage-1 cooldown 0.6-0.8 + freeze 0.8 | -6.2 | -5.8 | -6.0 |
| cooldown 0.55-0.75 + freeze 0.75 | -7.7 | -7.8 | -7.7 |
| freeze 0.8 + depth-2 stage 2, 8.25 ep | -13.1 | -14.4 | -13.7 |
| depth-2 stage 1, 8.0 ep | -5.6 | -5.8 | -5.7 |
