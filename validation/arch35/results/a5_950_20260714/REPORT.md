# A5 GammaAddRmsNorm arch35 验证报告（纠正版）

## 关键纠正

本报告纠正初版中的以下问题：
1. **xllm-ops kernel 现在可独立构建**：修复了 platform.h 对 ops::IsSame 的依赖（删除死代码 IsSupportAtomicAddTypeSIMD）、cust_opmaster/cust_proto 缺少 libops_base 链接、rms_norm_regbase_common.h 多余的 kernel_utils.h include。xllm-ops `build.sh --compute-unit=ascend950` 现在完整构建 kernel+host+op-api+安装包，退出码 0。
2. **精度判定如实**：ATK 精度 verdict = Failed（206/240 通过，34 失败），不是"0 失败"。34 个失败全部是半精度 MARE 超 1/128 阈值。
3. **验证用 xllm-ops 自建包**：direct_aclnn_bench 和 ATK smoke 均使用 xllm-ops 自建的 `custom_xllm_math` vendor 包验证通过。

## 环境
- 设备: Ascend950PR (A5), NPU 4, NpuArch 3510, soc Ascend950PR_9579
- CANN: 9.1.0-beta.1, Driver 25.7.rc1, ATK 26.5.14
- xllm-ops commit: eb016ba5 (基线) + 修复 commit
- 构建命令: `bash build.sh --build-dir=/workspace/build_gamma_a5 --op-name=gamma_add_rms_norm --compute-unit=ascend950`
- 构建退出码: 0 (SUCCESS)
- 安装: /usr/local/Ascend/cann-9.1.0-beta.1/opp/vendors/custom_xllm_math/
- ACLNN 符号: aclnnGammaAddRmsNorm + aclnnGammaAddRmsNormGetWorkspaceSize 均存在
- arch35 kernel: 3 个 .o (fp16/bf16/fp32 tiling key variants)

## xllm-ops 自建包验证（direct_aclnn_bench smoke）
- FP16 4x2048: PASS, max_abs=0, chain_ratio=0.64x
- BF16 4x2048: PASS, max_abs=0
- FP32 4x2048: PASS, max_abs=0

## xllm-ops 自建包验证（ATK 6 dtype×offset smoke）
| dtype | offset | verdict |
|---|---|---|
| FP16 | true | Pass |
| FP16 | false | Pass |
| BF16 | true | Pass |
| BF16 | false | Pass |
| FP32 | true | Pass |
| FP32 | false | Pass |

## 完整 240-case ATK 精度（ops-nn 构建的 kernel，与 xllm-ops kernel 同源）
- 总用例: 240, 执行成功: 240, 执行失败: 0
- **ATK 精度 verdict: Failed**
- 精度全三输出通过: 206/240 (85.8%)
- FP32: 80/80 (100%), FP16: 67/80 (83.75%), BF16: 59/80 (73.75%)
- 34 失败: 全部半精度, MARE 0.0100~0.0238 超 1/128 阈值(0.009765625)
- y MARE: max=2.38e-2, mean=5.22e-3; rstd MARE: max=2.85e-3

## ATK 动态 shape 性能（30 shapes, ops-nn kernel）
- GammaAddRmsNorm(true) vs 旧链路: 30/30 faster, geomean 1.387x

## msprof device task duration（7 shapes, ops-nn kernel）
| shape | Add(us) | AddRmsNorm(us) | old_chain(us) | GammaAddRmsNorm(us) | speedup | reduction |
|---|---|---|---|---|---|---|
| 1x2048 | 1.552 | 1.971 | 3.523 | 1.999 | 1.762x | 43.25% |
| 4x2048 | 1.578 | 2.436 | 4.014 | 2.474 | 1.623x | 38.37% |
| 31x2048 | 1.582 | 5.080 | 6.662 | 5.095 | 1.308x | 23.52% |
| 1x5120 | 1.603 | 2.638 | 4.242 | 2.665 | 1.592x | 37.18% |
| 4x5120 | 1.600 | 2.729 | 4.329 | 2.763 | 1.567x | 36.17% |
| 31x5120 | 1.623 | 5.251 | 6.874 | 5.302 | 1.296x | 22.86% |
| 4x8192 | 1.655 | 3.153 | 4.808 | 3.140 | 1.531x | 34.68% |
- geomean: 1.517x, 7/7 faster

## Host UT
- 10/10 通过, key 1000/2000 验证 addGammaOffset==1

## 源码修复清单
1. build_aclnn.sh: +ascend950 SOC branch
2. build.sh: ascend950 in SUPPORT list, remove stale sed translation
3. CMakeLists.txt: ascend910_95→ascend950
4. const_var.py: ascend950→Ascend950PR_9599
5. platform.h: 自包含（删除依赖 ops::IsSame 的死代码, 去掉 kernel_utils.h include）
6. rms_norm_regbase_common.h: 去掉多余的 kernel_utils.h include
7. gamma regbase_common.h: 同上
8. symbol.cmake: cust_opmaster 链接 libops_base.so
9. custom_build.cmake: cust_opmaster + cust_proto 链接 libops_base.so
10. stub/op_tiling/CMakeLists.txt: optiling 链接 libops_base.so
11. Vendored rms_norm/norm_common kernel headers (genuine cross-op deps)
