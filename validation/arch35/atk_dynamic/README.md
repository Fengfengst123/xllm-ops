# Dynamic-shape ATK performance validation

This directory validates BF16 GammaAddRmsNorm performance with shapes generated
from the ATK YAML constraints. The Python generator does not select or replace
`x1.shape`; it only enforces the operator relationships:

- `x2.shape == x1.shape`
- `gamma.shape == [x1.shape[-1]]`
- matching input dtypes and BF16 `kernelType`

The generated GammaAddRmsNorm cases are converted into aligned AddRmsNorm and
`Add(gamma, 1) + AddRmsNorm` cases. All three paths therefore use identical
shapes, dtypes, ranges, and seeds.

Run the complete generation and two ATK comparisons with:

```bash
DEVICE=0 \
INSTALL_ROOT=/path/to/custom/op/install \
ST_DIR=/path/to/ops-nn/norm/gamma_add_rms_norm/tests/st/aclnnGammaAddRmsNorm \
./run_atk_perf.sh
```

The default sampling is 100 warmups, 20 measured iterations, and 10 ATK
performance repeats per case. See `RESULTS.md` and `performance_results.csv`.
