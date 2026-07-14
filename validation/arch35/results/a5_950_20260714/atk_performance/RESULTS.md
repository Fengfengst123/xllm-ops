# ATK dynamic-shape performance result

## Configuration

- Device: Ascend950PR (A5), physical NPU 4 (Ascend950PR_9579)
- ATK task: `performance_device`
- Data type: BF16
- Cases: 30 unique shapes generated from YAML constraints
- Sampling: `--performance_data 100,20,10`
- Candidate: `GammaAddRmsNorm(addGammaOffset=true)`
- Baselines: bare `AddRmsNorm`, and `Add(gamma, 1) + AddRmsNorm`

The shape generator keeps ATK's generated `x1.shape`; its Python constraint
only derives `x2.shape` and `gamma.shape` from `x1`. No complete input shape is
fixed in the generator.

## Summary

- Versus old chain: geometric-mean speedup 1.387x; candidate faster on 30/30 shapes.
- Versus bare AddRmsNorm: geometric-mean ratio 0.996x; candidate no slower on 15/30 shapes.
- Mean old standalone Add task: 1.487 us.
- ATK report verdict versus old chain: Pass; average device performance ratio 1.0826.
- ATK report verdict versus bare AddRmsNorm: Failed; average device performance ratio 0.9438.

For model-relevant cases with the normalized hidden dimension `D >= 1024`,
the candidate is faster than the old chain on all 6/6 generated shapes, with
a 1.262x geometric-mean speedup. For decode-like cases (`M <= 31`, `D >= 1024`),
it is faster on all 3/3 shapes, with a 1.459x geometric-mean speedup.

The ATK verdict uses ATK's own report aggregation. The geometric means above
are independently calculated from the mean task durations in every exported
`op_summary.csv`; old-chain time is `mean(Add) + mean(AddRmsNorm)`.

## Per-shape results

| ID | Shape | GammaAddRmsNorm | AddRmsNorm | Old Add | Old chain | Chain speedup | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `[31, 2]` | 5.197 us | 4.963 us | 1.381 us | 6.419 us | 1.235x | 19.04% |
| 1 | `[31, 2]` | 4.998 us | 4.942 us | 1.410 us | 6.419 us | 1.284x | 22.15% |
| 2 | `[1, 64, 2]` | 3.733 us | 3.509 us | 1.380 us | 4.939 us | 1.323x | 24.41% |
| 3 | `[64, 2]` | 3.741 us | 3.558 us | 1.400 us | 4.931 us | 1.318x | 24.13% |
| 4 | `[31, 16]` | 4.938 us | 5.112 us | 1.381 us | 6.319 us | 1.280x | 21.86% |
| 5 | `[1024, 2]` | 2.786 us | 2.967 us | 1.367 us | 4.243 us | 1.523x | 34.35% |
| 6 | `[4, 4, 256]` | 3.515 us | 3.490 us | 1.414 us | 4.844 us | 1.378x | 27.44% |
| 7 | `[8, 512]` | 2.879 us | 2.840 us | 1.479 us | 4.231 us | 1.470x | 31.95% |
| 8 | `[1, 5120]` | 2.661 us | 2.413 us | 1.672 us | 4.042 us | 1.519x | 34.17% |
| 9 | `[64, 8, 16]` | 2.735 us | 2.873 us | 1.434 us | 4.261 us | 1.558x | 35.80% |
| 10 | `[64, 128]` | 3.708 us | 3.448 us | 1.442 us | 4.960 us | 1.337x | 25.23% |
| 11 | `[2, 4096]` | 2.479 us | 2.382 us | 1.675 us | 4.080 us | 1.646x | 39.24% |
| 12 | `[8192, 2]` | 3.608 us | 3.735 us | 1.439 us | 5.347 us | 1.482x | 32.53% |
| 13 | `[128, 128]` | 3.427 us | 3.387 us | 1.413 us | 4.787 us | 1.397x | 28.40% |
| 14 | `[64, 31, 16]` | 2.777 us | 2.938 us | 1.370 us | 4.336 us | 1.561x | 35.94% |
| 15 | `[1024, 31]` | 2.748 us | 2.851 us | 1.363 us | 4.259 us | 1.550x | 35.47% |
| 16 | `[1, 16, 2048]` | 3.727 us | 3.506 us | 1.667 us | 5.360 us | 1.438x | 30.47% |
| 17 | `[1, 5120, 8]` | 3.212 us | 3.358 us | 1.375 us | 4.733 us | 1.473x | 32.12% |
| 18 | `[7168, 8, 1]` | 9.654 us | 9.675 us | 1.462 us | 11.236 us | 1.164x | 14.08% |
| 19 | `[31, 2048]` | 5.115 us | 5.154 us | 1.644 us | 6.901 us | 1.349x | 25.88% |
| 20 | `[31, 31, 128]` | 2.792 us | 2.993 us | 1.392 us | 4.364 us | 1.563x | 36.02% |
| 21 | `[4096, 31]` | 3.180 us | 3.367 us | 1.363 us | 4.664 us | 1.466x | 31.81% |
| 22 | `[4, 31, 1024]` | 3.566 us | 3.647 us | 1.612 us | 5.078 us | 1.424x | 29.78% |
| 23 | `[31, 2, 4096]` | 4.319 us | 4.139 us | 1.688 us | 5.832 us | 1.350x | 25.95% |
| 24 | `[4, 256, 256]` | 3.638 us | 3.299 us | 1.379 us | 4.673 us | 1.285x | 22.15% |
| 25 | `[512, 2, 256]` | 3.623 us | 3.326 us | 1.413 us | 4.772 us | 1.317x | 24.08% |
| 26 | `[64, 8192]` | 6.075 us | 5.983 us | 1.857 us | 8.132 us | 1.339x | 25.30% |
| 27 | `[1, 256, 4096]` | 5.139 us | 5.185 us | 1.696 us | 7.076 us | 1.377x | 27.37% |
| 28 | `[31, 5120, 8]` | 22.099 us | 22.022 us | 1.236 us | 23.524 us | 1.064x | 6.06% |
| 29 | `[256, 8192]` | 6.866 us | 7.200 us | 1.796 us | 9.020 us | 1.314x | 23.89% |
