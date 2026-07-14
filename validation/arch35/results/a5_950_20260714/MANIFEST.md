# A5 Validation Results Manifest

## Result directory
`validation/arch35/results/a5_950_20260714/`

## Files (all logs are from xllm-ops self-built package unless noted)

| File | Purpose |
|---|---|
| REPORT.md | Final validation report |
| MANIFEST.md | This file |
| environment.txt | Device & software versions |
| build/build_gamma_a5.log | xllm-ops build log (exit 0, kernel built) |
| build/build_opsnn_gamma_a5.log | ops-nn build log (for comparison) |
| build/xllm_ops_build_status.md | Build status & fix details |
| build/sha256_manifest.txt | SHA256 of .run, kernel .o, libcust_opapi.so |
| build/build_artifacts.txt | Build artifact paths |
| build/install_artifacts.txt | Install artifact paths |
| build/direct_smoke_xllmops_built_4x2048.log | xllm-ops kernel smoke test |
| build/direct_smoke_opsnn_4x2048.log | ops-nn kernel smoke (comparison) |
| build/direct_smoke_opsnn_4x8192.log | ops-nn kernel 8192 smoke |
| host_ut/gamma_arch35_host_ut.log | Host UT log (10/10 pass) |
| host_ut/host_ut_summary.txt | UT summary |
| atk_accuracy/atk_accuracy_a5.log | 240-case ATK log (xllm-ops package) |
| atk_accuracy/atk_accuracy_detail.csv | Per-case precision detail |
| atk_accuracy/atk_accuracy_summary.md | Accuracy summary |
| atk_accuracy/atk_accuracy_reports.xlsx | ATK Excel report |
| atk_performance/RESULTS.md | ATK perf summary (30/30, 1.387x) |
| atk_performance/performance_results.csv | Per-shape perf data |
| atk_performance/atk_dynamic_perf_summary.log | Perf summary script output |
| msprof/msperf_device_summary.csv | Device task duration (4 shapes, xllm-ops package) |
| msprof/op_summary_*.csv | Per-shape op_summary (4 files) |
| patches/*.diff | All fix patches with root-cause READMEs |
