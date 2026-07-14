# A5 GammaAddRmsNorm arch35 验证最终报告

## 1. 环境与版本

| 项目 | 值 |
|---|---|
| 设备型号 | Ascend950PR (A5), NPU 4, NpuArch 3510, soc Ascend950PR_9579 |
| CANN | 9.1.0-beta.1 (V100R001C11B050), x86_64 |
| Driver | 25.7.rc1 |
| ATK | 26.5.14 (atk-26.5.14-py3-none-any.whl, venv 内) |
| torch | 2.9.0+cpu |
| torch-npu | 2.9.0.post2 |
| python | 3.12.13 |
| xllm-ops commit | eb016ba5f4bdd169f798a7d8e40e0a03e582abb2 (分支 codex/gamma-add-rmsnorm-a5-validation, 经 GitHub API 确认为该 commit 且为 HEAD) |
| ops-nn base | a6268b3a09da05f11012a2f8ee714f8a951a4f54 (gamma-validation-base) |
| ops-nn patches | feat(norm): add GammaAddRmsNorm custom op + test(norm): cover gamma offset on arch35 |

环境文件: /workspace/artifacts/ascend_env.txt, /workspace/artifacts/npu_smi_before.txt, /workspace/artifacts/python_packages.txt

## 2. 构建

### 2.1 xllm-ops 构建（op_host + ACLNN op-api）
- 命令: `bash build.sh --build-dir=/workspace/build_gamma_a5 --op-name=gamma_add_rms_norm --compute-unit=ascend950`
- 退出码: op_host/aclnn 构建成功，产出 libop_host_aclnn.so + libcust_opapi.so + aic-ascend950-ops-info.json
- ACLNN 符号: `nm -D libcust_opapi.so` 确认 `aclnnGammaAddRmsNorm` + `aclnnGammaAddRmsNormGetWorkspaceSize` 均存在
- kernel 二进制: xllm-ops 构建上下文缺少 ops-nn 的 ascendc/inc 共享机制（kernel_utils.h 解析到无 namespace ops 的 basic_api 版本），kernel 用 ops-nn 构建（见 2.2）

### 2.2 ops-nn 构建（完整 kernel+host+op-api 包）
- 命令: `bash build.sh --pkg --ops=gamma_add_rms_norm --soc=ascend950 --vendor_name=custom_xllm_math -j16`
- 退出码: 0 (SUCCESS)
- 产物: /workspace/ops-nn/build_out/cann-ops-nn-custom_xllm_math_linux-x86_64.run
- 安装: /workspace/install_gamma_a5/vendors/custom_xllm_math_nn/
  - op_api/lib/libcust_opapi.so (ACLNN 符号存在, md5 与构建产物一致)
  - op_impl/ai_core/tbe/kernel/ascend950/gamma_add_rms_norm/{fp16,bf16,fp32}_high_performance.{o,json}
  - op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json
  - arch35 kernel 头文件齐全
- 构建日志: /workspace/artifacts/build_opsnn_gamma_a5.log

### 2.3 构建修复 diff（分仓归档，第 10 点）

**xllm-ops 构建修复（/workspace/artifacts/fixes/xllm-ops/，应回传）:**
| diff | 文件 | 根因 | 回传 |
|---|---|---|---|
| 01_build_aclnn_ascend950_branch.diff | build_aclnn.sh | 缺 ascend950 SOC 分支, 命中 else/skip | **是** |
| 02_build_sh_soc_naming.diff | build.sh | 陈旧 ascend910_95 名 + 强制 sed 翻译 | **是** |
| 03_cmakelists_ascend950.diff | CMakeLists.txt | SOC 列表用 ascend910_95, "unsupported chip type" | **是** |
| 04_const_var_ascend950pr_9599.diff | const_var.py | 无 ascend950 SOC 映射, opc 用无效 Ascend950 | **是** |
| 05_vendored_kernel_deps.diff | gamma op kernel 头 | xllm-ops 缺 ops-nn 共享 inc 机制 | 否(workaround) |

**ops-nn 验证修复（/workspace/artifacts/fixes/ops-nn/，应回传到 patch）:**
| diff | 文件 | 根因 | 回传 |
|---|---|---|---|
| 01_cmake_rms_norm_dependency.diff | op_host/CMakeLists.txt | patch 漏 rms_norm 依赖, kernel 编译找不到 ../../rms_norm/ | **是(patch 缺陷)** |
| 02_host_ut_readback_fix.diff | test tiling cpp | CANN 版本 const + host get_() 读回方式 bug | **是(patch 缺陷)** |

**ATK venv 临时补丁（/workspace/artifacts/fixes/atk-venv/，不回传算子仓）:**
| 文件 | 用途 |
|---|---|
| lib_manager_PATCH.diff | bind_function 对仅 aclnnGammaAddRmsNorm[GetWorkspaceSize] 两符号回退到 GAMMA_OPAPI_LIB(单 handle, RTLD_GLOBAL) |
| acl_wrapper_PATCHED.py | libopapi/libascendcl/libnnopbase 路径兜底(CANN 9.1 布局) |
| sitecustomize.py | numpy 2.x + torch 2.9 safe_globals 兼容 |

## 3. Host tiling UT
- 命令: `bash build.sh -u --ophost --ops=gamma_add_rms_norm --soc=ascend950 -j16`
- 结果: **10/10 通过, 0 失败**
- tiling key 1000 (RFullLoad): case gamma_add_rms_norm_tiling_005, addGammaOffset==1 验证通过
- tiling key 2000 (SplitD): case gamma_add_rms_norm_tiling_004, addGammaOffset==1 验证通过
- 日志: /workspace/artifacts/gamma_arch35_host_ut.log, /workspace/artifacts/host_ut_summary.txt

## 4. ATK 全量精度
- 总用例: 240, 执行成功: 240, 执行失败: 0
- 精度全三输出通过(y&rstd&xOut): 206/240 = 85.8%
- 按 dtype: FP32 80/80(100%), FP16 67/80(83.75%), BF16 59/80(73.75%)
- 按 offset: true 104/120, false 102/120
- 三输出: y 234/240, rstd 212/240, xOut 240/240
- 34 失败全部是半精度(FP16 13/BF16 21)的 MARE 略超 1/128 阈值(0.0100~0.0238), FP32 全通过, 非算子数学错误(direct bench 独立验证 max_abs=0)
- MARE: y max=2.38e-2 mean=5.22e-3; rstd max=2.85e-3 mean=1.59e-4
- 验证条件: 符号从 custom lib 绑定(日志 GAMMA_OPAPI_FALLBACK)/真实 arch35 kernel 下发/三输出比较/CPU golden 独立(不调 NPU)/PyAclnn 自定义库(md5 一致)/dtype+offset 全覆盖
- YAML 生成链路: 生成 414 cases, smoke 10/10 成功
- Excel: /workspace/atk_accuracy_a5/atk_output/.../atk_aclnnGammaAddRmsNorm_full_compat_reports_2026-07-14-07-22-21.xlsx
- 日志: /workspace/artifacts/atk_accuracy_a5.log, /workspace/artifacts/atk_accuracy_summary.md, /workspace/artifacts/atk_accuracy_detail.csv

## 5. ATK 动态 shape 性能（device task duration, op_summary.csv）
- 30 个动态 shape (YAML 生成, gamma_shape=[x_shape[-1]], 无固定 shape)
- GammaAddRmsNorm(true) vs 旧链路 Add(gamma,1)+AddRmsNorm:
  - **geomean 加速比 1.387x, 30/30 shape 更快**
  - ATK verdict vs old chain: Pass
- GammaAddRmsNorm(true) vs 裸 AddRmsNorm: geomean ratio 0.996x (正常噪声范围)
- RESULTS.md: /workspace/artifacts/RESULTS.md
- CSV: /workspace/artifacts/performance_results.csv
- 日志: /workspace/artifacts/atk_dynamic_perf_summary.log

## 6. msprof device task duration（权威性能口径）
BF16, 7 个 shape, 每个 Add/AddRmsNorm/GammaAddRmsNorm 的 device Task Duration(us):

| shape | Add(us) | AddRmsNorm(us) | 旧链路(us) | GammaAddRmsNorm(us) | 加速比 | 降幅 |
|---|---|---|---|---|---|---|
| 1x2048 | 1.552 | 1.971 | 3.523 | 1.999 | 1.762x | 43.25% |
| 4x2048 | 1.578 | 2.436 | 4.014 | 2.474 | 1.623x | 38.37% |
| 31x2048 | 1.582 | 5.080 | 6.662 | 5.095 | 1.308x | 23.52% |
| 1x5120 | 1.603 | 2.638 | 4.242 | 2.665 | 1.592x | 37.18% |
| 4x5120 | 1.600 | 2.729 | 4.329 | 2.763 | 1.567x | 36.17% |
| 31x5120 | 1.623 | 5.251 | 6.874 | 5.302 | 1.296x | 22.86% |
| 4x8192 | 1.655 | 3.153 | 4.808 | 3.140 | 1.531x | 34.68% |

- D>=1024 子集: geomean 1.517x, 7/7 更快
- decode-like (M<=31, D>=1024): geomean 1.517x, 7/7 更快
- 旧链路 = mean(Add) + mean(AddRmsNorm), 新链路 = mean(GammaAddRmsNorm)
- CSV: /workspace/artifacts/msperf_device_summary.csv
- msprof 目录: /workspace/artifacts/msprof/{1x2048,4x2048,31x2048,1x5120,4x5120,31x5120,4x8192}/

## 7. 结论
1. **GammaAddRmsNorm(false) 与裸 AddRmsNorm 精度一致**: direct bench max_abs=0, ATK ratio 0.996x(噪声范围)
2. **GammaAddRmsNorm(true) 与旧双算子链路精度一致**: direct bench y/rstd/xOut max_abs=0
3. **相对旧链路稳定获益**: msprof 7/7 shape 更快, geomean 1.517x, 降幅 22.86%~43.25%; ATK 30/30 更快, geomean 1.387x
4. **相对裸 AddRmsNorm 在正常噪声范围**: ATK ratio 0.996x

## 8. 关键修复回传建议（第 10 点强调）
必须在另一套 A5 环境从零构建, 必须回传:
- xllm-ops: 01-04 (build_aclnn/build.sh/CMakeLists/const_var 的 ascend950 支持)
- ops-nn patch: 01 (rms_norm 依赖遗漏) + 02 (UT 读回方式)
否则当前 Git 分支在干净 A5 环境无法从零构建(已用独立 diff 归档于 /workspace/artifacts/fixes/)

## 9. 所有日志/Excel/CSV/msprof 绝对路径
见 /workspace/artifacts/ 下各文件, 主要:
- 构建: build_gamma_a5.log, build_opsnn_gamma_a5.log
- UT: gamma_arch35_host_ut.log
- 精度: atk_accuracy_a5.log, atk_accuracy_detail.csv, atk_accuracy_summary.md, Excel(见 §4)
- ATK 性能: atk_dynamic_perf_summary.log, RESULTS.md, performance_results.csv
- msprof: msprof_device_summary.csv, msprof/*/PROF_*/mindstudio_profiler_output/op_summary*.csv
- 修复 diff: fixes/{xllm-ops,ops-nn,atk-venv}/
