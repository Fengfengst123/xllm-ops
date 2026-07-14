# A5 GammaAddRmsNorm arch35 验证报告

## 摘要

xllm-ops 独立构建 ascend950 算子包（kernel + host + op-api），在 A5 (Ascend950PR) 上完成验证。

| 验证项 | 结果 |
|---|---|
| xllm-ops 构建 | exit 0, SUCCESS (kernel 3个.o + ACLNN 符号 + .run 包) |
| Host UT | 10/10 通过 (key 1000/2000, addGammaOffset==1) |
| ATK 精度 (240-case) | 240 执行成功, 0 执行失败; **精度 verdict: Failed** (206/240 通过, 34 半精度超 1/128 阈值) |
| ATK 动态性能 (30 shape) | 30/30 faster, geomean 1.387x |
| msprof device (4 shape, xllm-ops 包) | 4/4 faster, geomean 1.513x |
| direct_aclnn_bench | FP16/BF16/FP32 PASS, max_abs=0 |

## 环境
- 设备: Ascend950PR (A5), NPU 4, NpuArch 3510, soc Ascend950PR_9579
- CANN: 9.1.0-beta.1, Driver 25.7.rc1, ATK 26.5.14
- xllm-ops 基线 commit: eb016ba5f4bdd169f798a7d8e40e0a03e582abb2
- 构建命令: `bash build.sh --build-dir=/workspace/build_gamma_a5 --op-name=gamma_add_rms_norm --compute-unit=ascend950`
- 构建退出码: 0
- .run SHA256: 8f1a66077fd3f41966b4a621b3465da1e992039e0a792000d051d5bedd20ecb8

## 精度详情 (xllm-ops 自建包, 240-case ATK)

- 总用例: 240, 执行成功: 240, 执行失败: 0
- **ATK 精度 verdict: Failed**
- 精度全三输出通过: 206/240 (85.8%)
- 三输出: y 234/240, rstd 212/240, xOut 240/240

### 按 dtype × addGammaOffset
| dtype | offset | total | all_pass |
|---|---|---|---|
| FP32 | true | 40 | 40 |
| FP32 | false | 40 | 40 |
| FP16 | true | 40 | 35 |
| FP16 | false | 40 | 32 |
| BF16 | true | 40 | 29 |
| BF16 | false | 40 | 30 |

### 34 失败 case 分析
- 全部半精度 (FP16 13 + BF16 21), FP32 零失败
- 失败原因: MARE 超 ATK 默认阈值 1/128 (0.009765625), 实际值 0.0100~0.0238
- y MARE: max=2.38e-2, mean=5.22e-3; rstd MARE: max=2.85e-3, mean=1.59e-4
- 非算子 bug: direct_aclnn_bench 验证 GammaAddRmsNorm(true) 与旧链路 Add+AddRmsNorm 在相同输入上 max_abs=0; FP32 全通过

## 性能 (msprof device task duration, xllm-ops 自建包, BF16)
| shape | Add(us) | AddRmsNorm(us) | old_chain(us) | GammaAddRmsNorm(us) | speedup | reduction |
|---|---|---|---|---|---|---|
| 4x2048 | 1.622 | 2.491 | 4.113 | 2.509 | 1.640x | 39.01% |
| 31x2048 | 1.575 | 5.006 | 6.581 | 5.032 | 1.308x | 23.54% |
| 4x5120 | 1.629 | 2.720 | 4.349 | 2.753 | 1.580x | 36.71% |
| 4x8192 | 1.676 | 3.073 | 4.749 | 3.071 | 1.546x | 35.33% |
- geomean: 1.513x, 4/4 faster

## 源码修复清单 (11 项, 无算子数学改动)
1. build_aclnn.sh: +ascend950 SOC branch
2. build.sh: ascend950 in SUPPORT list, remove stale sed
3. CMakeLists.txt: ascend910_95→ascend950
4. const_var.py: ascend950→Ascend950PR_9599
5. platform.h: self-contained (remove ops::IsSame dead code, drop kernel_utils.h)
6. rms_norm_regbase_common.h: drop unused kernel_utils.h
7. gamma regbase_common.h: drop unused kernel_utils.h
8. symbol.cmake: cust_opmaster link libops_base.so
9. custom_build.cmake: cust_opmaster + cust_proto link libops_base.so
10. stub/op_tiling/CMakeLists.txt: optiling link libops_base.so
11. Vendored rms_norm/norm_common/platform.h kernel headers (genuine cross-op deps)
