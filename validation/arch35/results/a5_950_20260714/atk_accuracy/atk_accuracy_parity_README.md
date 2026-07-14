# ATK Accuracy Parity Report (240-case, same executor/golden/threshold)

## Method
Both chains ran through the **same ATK executor pipeline** with identical:
- CPU golden (same torch computation in old_chain_accuracy_executor.py)
- MARE aggregation and thresholds (ATK's built-in, per-output-type thresholds)
- Input data (same 240-case JSON, same random generation)

### Chain 1: GammaAddRmsNorm
- NPU: aclnnGammaAddRmsNorm (xllm-ops self-built package, vendor: custom_xllm_math)
- Plugin: executor_aclnnGammaAddRmsNorm.py (pyaclnn backend)
- Case JSON: atk_aclnnGammaAddRmsNorm_full_compat.json

### Chain 2: Old chain Add(gamma,1)+AddRmsNorm
- NPU: torch_npu.npu_add_rms_norm with gamma+1.0 pre-computed
- Plugin: old_chain_accuracy_executor.py (npu backend)
- Case JSON: old_chain_240_compat.json (same inputs, api_type changed)

## Result

| Chain | Pass | Fail | Verdict |
|---|---|---|---|
| GammaAddRmsNorm | 206/240 | 34 | Failed |
| Old chain | 206/240 | 34 | Failed |
| **Verdict match** | **240/240** | | |
| **Fail set identical** | **Yes** | | |

### By dtype
| dtype | Gamma pass | Old chain pass |
|---|---|---|
| FP16 | 67/80 | 67/80 |
| BF16 | 59/80 | 59/80 |
| FP32 | 80/80 | 80/80 |

### Fail set (identical for both chains)
Cases: 0, 2, 4, 12, 14, 15, 16, 17, 18, 19, 24, 25, 28, 29, 30, 31, 40, 41, 42, 43, 76, 77, 78, 79, 87, 100, 101, 102, 103, 135, 207, 217, 229, 231

## Conclusion
**ATK accuracy parity validated.** GammaAddRmsNorm and the old chain
Add(gamma,1)+AddRmsNorm produce **identical verdicts and fail sets** across
all 240 cases when evaluated through the same ATK executor, CPU golden,
MARE algorithm, and thresholds. The 34 failures are inherent to half-precision
rounding and affect both chains equally.

This does NOT mean "240/240 passed" — the ATK official result remains
206/240, Failed. It means the two chains are accuracy-equivalent.

## Files
- `atk_accuracy_parity_240.csv`: per-case verdict/MARE comparison
- `old_chain_accuracy_executor.py`: old chain ATK executor plugin
- `old_chain_240_compat.json`: 240-case JSON for old chain
- `nodes_npu_cpu.yaml`: nodes config for old chain run
- `atk_oldchain_240.log`: old chain 240-case execution log
- `oldchain_accuracy_reports.xlsx`: old chain ATK Excel report
- `atk_accuracy_reports.xlsx`: GammaAddRmsNorm ATK Excel report
