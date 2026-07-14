# ATK 精度验证结果（xllm-ops 自建包, A5 / Ascend950PR_9579）

## 包信息
- vendor: custom_xllm_math (xllm-ops 自建)
- libcust_opapi.so SHA256: 1fcb24cac0f225a494c5fe4a00f1d4a56cc36a58e23d5971f5c396365f9a10b0
- 运行时间: 2026-07-14T14:02:11Z

## 总体
- 总用例数: 240
- 执行成功: 240
- 执行失败: 0
- 精度全三输出通过(y & rstd & xOut): 206/240 = 85.8%
- ATK 聚合判定: Failed

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

## 34 失败 case
- 全部半精度 (FP16 13 + BF16 21), FP32 零失败
- MARE 超 1/128 阈值 (0.009765625), 实际 0.0100~0.0238
- y MARE: max=2.38e-2, mean=5.22e-3; rstd MARE: max=2.85e-3, mean=1.59e-4
- 非算子 bug: direct_aclnn_bench max_abs=0, FP32 全通过

## 产物
- Excel: atk_accuracy_reports.xlsx
- 日志: atk_accuracy_a5.log (含 GAMMA_OPAPI_FALLBACK 符号绑定 + kernel 下发证据)
- 明细: atk_accuracy_detail.csv
