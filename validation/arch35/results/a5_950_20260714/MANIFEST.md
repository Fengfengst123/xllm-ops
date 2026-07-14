# A5 Validation Results Manifest

## Result directory
`validation/arch35/results/a5_950_20260714/`

## Files

| File | Source (original absolute path) | Purpose | Size |
|---|---|---|---|
| REPORT.md | /workspace/artifacts/FINAL_REPORT.md | Final validation report | ~8KB |
| MANIFEST.md | (this file) | File manifest | ~4KB |
| environment.txt | /workspace/artifacts/{npu_smi_before,ascend_env,python_packages}.txt | Device & software versions | ~4KB |
| build/build_opsnn_gamma_a5.log | /workspace/artifacts/build_opsnn_gamma_a5.log | ops-nn build log (exit 0) | ~30KB |
| build/build_gamma_a5.log | /workspace/artifacts/build_gamma_a5.log | xllm-ops build log | ~350KB |
| build/xllm_ops_build_status.md | /workspace/artifacts/xllm_ops_build_status.md | xllm-ops build status & fixes | ~3KB |
| build/build_artifacts.txt | /workspace/artifacts/build_artifacts.txt | Build artifact paths | ~2KB |
| build/install_artifacts.txt | /workspace/artifacts/install_artifacts.txt | Install artifact paths | ~1KB |
| host_ut/gamma_arch35_host_ut.log | /workspace/artifacts/gamma_arch35_host_ut.log | Host UT log (10/10 pass) | ~20KB |
| host_ut/host_ut_summary.txt | /workspace/artifacts/host_ut_summary.txt | UT summary | ~2KB |
| atk_accuracy/atk_accuracy_summary.md | /workspace/artifacts/atk_accuracy_summary.md | Accuracy summary (240/0, 206/240 pass) | ~3KB |
| atk_accuracy/atk_accuracy_detail.csv | /workspace/artifacts/atk_accuracy_detail.csv | Per-case accuracy detail | ~30KB |
| atk_accuracy/atk_accuracy_a5.log | /workspace/artifacts/atk_accuracy_a5.log | Full 240-case ATK log | ~500KB |
| atk_accuracy/atk_accuracy_reports.xlsx | /workspace/atk_accuracy_a5/.../*.xlsx | ATK Excel report | ~50KB |
| atk_performance/RESULTS.md | /workspace/artifacts/RESULTS.md | ATK perf summary (30/30, 1.387x) | ~5KB |
| atk_performance/performance_results.csv | /workspace/artifacts/performance_results.csv | Per-shape perf data | ~3KB |
| atk_performance/atk_dynamic_perf_summary.log | /workspace/artifacts/atk_dynamic_perf_summary.log | Perf summary script output | ~5KB |
| msprof/msperf_device_summary.csv | /workspace/artifacts/msperf_device_summary.csv | Device task duration summary (7 shapes) | ~1KB |
| msprof/op_summary_*.csv | /workspace/artifacts/msprof/*/.../op_summary*.csv | Per-shape op_summary (7 files) | ~5KB each |
| patches/01_build_aclnn_ascend950_branch.diff | /workspace/artifacts/fixes/xllm-ops/ | build_aclnn ascend950 branch fix | ~1KB |
| patches/02_build_sh_soc_naming.diff | /workspace/artifacts/fixes/xllm-ops/ | SOC naming fix | ~1KB |
| patches/03_cmakelists_ascend950.diff | /workspace/artifacts/fixes/xllm-ops/ | CMakeLists ascend950 fix | ~2KB |
| patches/04_const_var_ascend950pr_9599.diff | /workspace/artifacts/fixes/xllm-ops/ | const_var Ascend950PR_9599 fix | ~1KB |
| patches/05_vendored_kernel_deps.diff | /workspace/artifacts/fixes/xllm-ops/ | Vendored kernel deps workaround | ~140KB |
| patches/01_cmake_rms_norm_dependency.diff | /workspace/artifacts/fixes/ops-nn/ | rms_norm dependency fix (patch defect) | ~1KB |
| patches/02_host_ut_readback_fix.diff | /workspace/artifacts/fixes/ops-nn/ | UT readback fix (CANN compat) | ~3KB |
| patches/lib_manager_PATCH.diff | /workspace/artifacts/fixes/atk-venv/ | ATK bind_function fallback (venv-only) | ~3KB |
| patches/*_fixes_README.md | /workspace/artifacts/fixes/*/README.md | Root cause + backport guidance per repo | ~3KB each |

## Uncommitted large files (kept in /workspace/artifacts only)

| Path | Size | SHA256 | Reason |
|---|---|---|---|
| /workspace/venv/ | 1.1GB | n/a | Isolated Python venv (rebuildable) |
| /workspace/build_gamma_a5/ | 116MB | n/a | Build intermediate (rebuildable) |
| /workspace/ops-nn/build/ | ~600MB | n/a | ops-nn build (rebuildable) |
| /workspace/install_gamma_a5/ | 6.4MB | n/a | Installed .run package (rebuildable) |
| /workspace/artifacts/msprof/*/PROF_*/ | ~50MB total | n/a | Full msprof raw timeline (op_summary.csv extracted instead) |
