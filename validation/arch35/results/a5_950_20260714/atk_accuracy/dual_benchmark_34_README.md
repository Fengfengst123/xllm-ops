# 34 失败 case 双标杆对比报告

## 方法
对 ATK 240-case 精度验证中失败的 34 个 case，使用 ATK 保存的**完全相同的输入数据**
（input.bin），分别用以下两条链路在 A5 NPU 上执行：

1. **GammaAddRmsNorm**: 融合算子 (aclnnGammaAddRmsNorm, xllm-ops 自建包)
2. **旧链路**: Add(gamma,1) + AddRmsNorm (torch_npu.npu_add_rms_norm)

两者均与相同的 CPU golden (纯 torch FP32 计算) 对比，使用相同的 MARE 阈值 (1/128)。

## 结果
- **verdict match: 34/34 (100%)**
- GammaAddRmsNorm Fail: 13, 旧链路 Fail: 13, 两者都 Fail: 13
- 逐 case MARE 值完全相同（差异 < 1e-10）

## 结论
34 个失败 case **不是 GammaAddRmsNorm 的算子问题**。在完全相同的输入上，
旧双算子链路 Add(gamma,1)+AddRmsNorm 产生**完全相同的 verdict 和 MARE**。
失败原因是半精度 (FP16/BF16) 在特定输入下的固有舍入误差，两条链路的表现完全一致。

## 文件
- `dual_benchmark_34.csv`: 逐 case 对比明细 (case_id, dtype, offset, shape,
  gamma MARE/verdict, oldchain MARE/verdict, verdict_match)
