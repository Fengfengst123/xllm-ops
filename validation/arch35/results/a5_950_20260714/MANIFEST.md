# A5 Validation Results Manifest

## Result directory
`validation/arch35/results/a5_950_20260714/`

## Package provenance
All logs, CSVs, and Excel in this directory are generated from the **xllm-ops
self-built package** (vendor: custom_xllm_math) unless explicitly marked as
legacy_opsnn. The xllm-ops build command was:
```
bash build.sh --build-dir=/workspace/build_gamma_a5 --op-name=gamma_add_rms_norm --compute-unit=ascend950
```
Build exit code: 0. See build/sha256_manifest.txt for artifact hashes.

## Files

| File | Purpose | Source |
|---|---|---|
| REPORT.md | Final validation report | xllm-ops package |
| MANIFEST.md | This file | - |
| environment.txt | Device & software versions | - |
| build/build_gamma_a5.log | xllm-ops build log (exit 0, kernel built) | xllm-ops build |
| build/build_opsnn_gamma_a5.log | ops-nn build log (comparison only) | ops-nn build |
| build/xllm_ops_build_status.md | Build status & fix details | - |
| build/sha256_manifest.txt | SHA256 of .run, kernel .o, libcust_opapi.so | xllm-ops build |
| build/build_artifacts.txt | Build artifact paths | - |
| build/install_artifacts.txt | Install artifact paths | - |
| build/direct_smoke_xllmops_built_4x2048.log | xllm-ops kernel smoke test (3 dtype PASS) | xllm-ops package |
| build/direct_smoke_opsnn_4x2048.log | ops-nn kernel smoke (comparison) | ops-nn package |
| build/direct_smoke_opsnn_4x8192.log | ops-nn kernel 8192 smoke (comparison) | ops-nn package |
| host_ut/gamma_arch35_host_ut.log | Host UT log (10/10 pass) | ops-nn UT |
| host_ut/host_ut_summary.txt | UT summary | - |
| atk_accuracy/atk_accuracy_a5.log | 240-case ATK log (xllm-ops package, includes GAMMA_OPAPI_FALLBACK + kernel dispatch evidence) | xllm-ops package |
| atk_accuracy/atk_accuracy_detail.csv | Per-case precision detail (xllm-ops package, 240 rows) | xllm-ops package |
| atk_accuracy/atk_accuracy_summary.md | Accuracy summary (xllm-ops package) | xllm-ops package |
| atk_accuracy/atk_accuracy_reports.xlsx | ATK Excel report (xllm-ops package) | xllm-ops package |
| atk_performance/RESULTS.md | ATK perf summary (30/30, 1.387x) | ops-nn package |
| atk_performance/performance_results.csv | Per-shape perf data | ops-nn package |
| atk_performance/atk_dynamic_perf_summary.log | Perf summary script output | ops-nn package |
| msprof/msperf_device_summary.csv | Device task duration (4 shapes, xllm-ops package, geomean 1.513x) | xllm-ops package |
| msprof/op_summary_4x2048.csv | op_summary for 4x2048 | xllm-ops package |
| msprof/op_summary_31x2048.csv | op_summary for 31x2048 | xllm-ops package |
| msprof/op_summary_4x5120.csv | op_summary for 4x5120 | xllm-ops package |
| msprof/op_summary_4x8192.csv | op_summary for 4x8192 | xllm-ops package |
| msprof/legacy_opsnn/op_summary_1x2048.csv | op_summary for 1x2048 (legacy) | ops-nn package |
| msprof/legacy_opsnn/op_summary_1x5120.csv | op_summary for 1x5120 (legacy) | ops-nn package |
| msprof/legacy_opsnn/op_summary_31x5120.csv | op_summary for 31x5120 (legacy) | ops-nn package |
| patches/*.diff | All fix patches with root-cause READMEs | - |
| atk_accuracy/dual_benchmark_34.csv | 34-case dual benchmark (gamma vs old chain, same input) | xllm-ops package |
| atk_accuracy/dual_benchmark_34_README.md | Dual benchmark report (34/34 verdict match) | - |

## Notes
- atk_accuracy_a5.log contains GAMMA_OPAPI_FALLBACK line proving aclnnGammaAddRmsNorm
  symbols were bound from the xllm-ops self-built libcust_opapi.so (not CANN system libopapi.so).
- atk_accuracy_detail.csv was regenerated from the xllm-ops 240-case run Excel;
  its git blob differs from the initial ffd2a90 commit.
- msprof/ has 4 xllm-ops op_summary CSVs + 3 legacy ops-nn CSVs in legacy_opsnn/.
- ATK dynamic performance (30 shapes) was run with the ops-nn package; the op_summary
  evidence for xllm-ops is the 4-shape msprof above.
