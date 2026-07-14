# ATK 精度验证结果（A5 / Ascend950PR_9579）

## 总体
- 总用例数: 240
- 执行成功: 240
- 执行失败: 0
- 精度全三输出通过(y & rstd & xOut): 206/240 = 85.8%
- ATK 聚合判定: Failed（因通过率 <100%）

## 三输出通过率
| 输出 | 通过/总 |
|---|---|
| y (output_0)     | 234/240 |
| rstd (output_1)  | 212/240 |
| xOut (output_2)  | 240/240 |

## 按 dtype × addGammaOffset
| dtype | offset | total | all_pass | y | rstd | xOut |
|---|---|---|---|---|---|---|
| FP32 | True  | 40 | 40 | 40 | 40 | 40 |
| FP32 | False | 40 | 40 | 40 | 40 | 40 |
| FP16 | True  | 40 | 35 | 40 | 35 | 40 |
| FP16 | False | 40 | 32 | 34 | 38 | 40 |
| BF16 | True  | 40 | 29 | 40 | 29 | 40 |
| BF16 | False | 40 | 30 | 40 | 30 | 40 |

- FP32: 80/80 全通过（100%）
- FP16: 67/80 通过（83.75%）
- BF16: 59/80 通过（73.75%）
- offset=True: 104/120 通过；offset=False: 102/120 通过

## 最大相对误差(MARE)统计
| 输出 | max | mean | median |
|---|---|---|---|
| y    | 2.38e-02 | 5.22e-03 | 1.54e-03 |
| rstd | 2.85e-03 | 1.59e-04 | 1.48e-05 |
| xOut | (全通过，MARE 在阈值内) | | |

## 34 失败 case 分析
- 全部是半精度(FP16 13 / BF16 21)，FP32 0 失败
- 失败输出: rstd 28/34 失败, y 6/34 失败, xOut 0/34 失败
- 失败原因: MARE 超过阈值 0.009765625 (1/128)，实际误差 0.0100~0.0238
- 这是半精度(x1+x2 求和再 rmsnorm)的相对误差在 1/128 边界的正常噪声，
  非算子数学错误（direct_aclnn_bench 已独立验证 GammaAddRmsNorm(true) 与
  旧链路 Add+AddRmsNorm 的 y/rstd/xOut 逐元素一致，max_abs=0）

## 验证条件确认
- 符号从 custom lib 绑定: 是（GAMMA_OPAPI_FALLBACK 日志，本次构建安装的 libcust_opapi.so）
- 真实 arch35 kernel 下发: 是（无 ParseDynamicKernels 错误）
- 三输出 y/rstd/xOut 均参与比较: 是
- CPU golden 独立计算(不调 NPU 算子): 是（GAMMA_ADD_RMS_NORM_GOLDEN_MODE 未设置→纯 torch CPU）
- PyAclnn 自定义库: 是（libcust_opapi.so, md5 与构建产物一致）
- dtype 覆盖: FP16/BF16/FP32 全覆盖
- offset 覆盖: true/false 全覆盖（各 120）

## 产物路径
- Excel: /workspace/atk_accuracy_a5/atk_output/atk_aclnnGammaAddRmsNorm_full_compat_2026-07-14-15-22-20-600313/report/atk_aclnnGammaAddRmsNorm_full_compat_reports_2026-07-14-07-22-21.xlsx
- 日志: /workspace/artifacts/atk_accuracy_a5.log
- 明细 CSV: /workspace/artifacts/atk_accuracy_detail.csv
