# 34 失败 case 双标杆对比报告

## 方法
对 ATK 240-case 精度验证中失败的 34 个 case，使用 ATK 保存的**完全相同的输入数据**
（input.bin），分别用以下两条链路在 A5 NPU 上执行：

1. **GammaAddRmsNorm**: 融合算子 (aclnnGammaAddRmsNorm, xllm-ops 自建包)
2. **旧链路**: Add(gamma,1) + AddRmsNorm (torch_npu.npu_add_rms_norm)

两者均与 CPU golden (纯 torch FP32 计算) 对比，使用相同的 MARE 阈值 (1/128)。

## 核心结果
- **gamma vs oldchain verdict match: 34/34 (100%)**
- 逐 case MARE 值完全相同（差异 < 1e-10）
- 这证明两链路在相同输入上行为完全等价

## ATK 原始 verdict vs 双标杆 verdict 的差异说明

| 分类 | 数量 | 说明 |
|---|---|---|
| ATK Fail + 双标杆 Fail | 13 | 两处均判定失败 |
| ATK Fail + 双标杆 Pass | 21 | ATK 判定失败，双标杆判定通过 |

**差异根因：CPU golden 的 rstd 计算方式不同**

- **ATK executor** 的 CPU golden: rstd reshape 到 rstdShape（保留前几维，最后一维=1），
  MARE 在该 shape 上计算，分母是 golden_rstd，当 golden_rstd 接近 0 时 MARE 被放大
- **双标杆脚本** 的 CPU golden: rstd expand_as(x_sum)（广播到完整 shape），
  MARE 在完整 shape 上计算，分母相同但由于广播导致计算路径略有差异

这 21 个 case 的 MARE 都在 1/128 阈值边界附近（0.007~0.013），两种 CPU golden 计算
方式导致的微小 MARE 差异足以翻转 verdict。**这不是 NPU 算子实现的差异**——在双标杆中
gamma 和 oldchain 使用完全相同的 CPU golden，两者的 verdict 始终一致（34/34 MATCH）。

## 原始 ATK 失败输出分布
- rstd (output_1) 失败: 28/34
- y (output_0) 失败: 6/34
- xOut (output_2) 失败: 0/34

rstd 是主要失败输出，因为 rstd 的值域接近 0（1/sqrt(mean(x^2)+eps)），相对误差对分母敏感。

## 结论
1. 34 个失败 **不是 GammaAddRmsNorm 算子问题**：在完全相同的输入上，旧链路产生完全
   相同的 verdict 和 MARE（34/34 MATCH）。
2. 21 个 verdict 翻转是**双标杆脚本与 ATK executor 的 CPU golden 计算方式差异**导致
   的（rstd reshape vs expand_as），不是 NPU 算子差异。
3. 所有失败集中在半精度 (FP16/BF16) 的 rstd/y 输出，MARE 在 1/128 阈值边界附近，
   属于半精度固有舍入噪声。FP32 全 80/80 通过。

## 文件
- `dual_benchmark_34.csv`: 逐 case 对比明细 (case_id, dtype, offset, shape,
  gamma MARE/verdict, oldchain MARE/verdict, verdict_match)
