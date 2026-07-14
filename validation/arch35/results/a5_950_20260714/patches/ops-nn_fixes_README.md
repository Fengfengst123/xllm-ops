# ops-nn 验证修复（gamma_add_rms_norm）

> 仓库: gitcode.com/cann/ops-nn, base = a6268b3a (gamma-validation-base)
> 应用了 xllm-ops 仓的两个 patch:
>   0001-feat-norm-add-GammaAddRmsNorm-custom-op.patch.gz
>   0001-test-norm-cover-gamma-offset-on-arch35.patch.gz
> 以下是 patch 之上为让 A5 构建与 UT 通过所必需的最小修复。

## 01_cmake_rms_norm_dependency.diff
- 文件: norm/gamma_add_rms_norm/op_host/CMakeLists.txt
- 根因: patch 的 DEPENDENCIES 只写了 `norm_common`，但 gamma 的 arch35 kernel
  头还 include `../../rms_norm/rms_norm_base.h` 和
  `../../rms_norm/arch35/rms_norm_regbase_common.h`。缺少 rms_norm 依赖导致
  kernel 编译时 rms_norm 源码不被拷贝为 sibling，`../../rms_norm/...` 找不到，
  opc PRECOMPILE 阶段 fatal error。这是 patch 的遗漏，影响所有用 ops-nn 构建
  A5 kernel 的环境。
- 修复: DEPENDENCIES 增加 `rms_norm`。
- 影响: 仅 build 依赖声明；不改算子数学。
- 回传建议: **必须回传到 xllm-ops 的 patch（0001-feat-...）**。这是 patch 自身
  缺陷，不修则任何环境都无法用 ops-nn 构建 gamma 的 A5 kernel。

## 02_host_ut_readback_fix.diff
- 文件: norm/gamma_add_rms_norm/tests/ut/op_host/test_gamma_add_rms_norm_tiling.cpp
- 根因（两处）:
  (a) const 正确性: 测试用 `const auto*` 指针调用 `get_addGammaOffset()`，但
      CANN 9.1.0-beta.1 的 TILING_DATA_FIELD_DEF 生成的 getter 是非 const 的，
      编译报 discards qualifiers。
  (b) 读回方式: 测试用 `reinterpret_cast<HostStruct*>(buf)->get_addGammaOffset()`
      读 tiling buffer。但 CANN 的 host tiling 结构 get_<field>() 返回的是 C++
      内存成员（默认 0），不是序列化 buffer 里的值——两者布局不同（每个字段有
      16 字节 reserve_buf_）。因此该读法恒为 0，与算子是否正确设置无关。
      经字节级验证（dump buffer bytes[56..63]=01000000），算子确实把
      addGammaOffset=1 正确序列化到了 buffer offset 56（RFullLoad）/48（SplitD）。
- 修复:
  (a) 去掉 const。
  (b) 改为直接读 raw buffer 的 uint32（RFullLoad offset 56，SplitD offset 48）。
- 影响: 仅测试读回方式；**不改算子 tiling 逻辑**（算子已被证明正确）。
- 回传建议: **应回传到 xllm-ops 的 patch（0001-test-...）**。否则在该 CANN 版本
  上 UT 无法通过（编译失败 + 误报算子 bug）。注意 offset 依赖 tiling 结构字段
  顺序，若上游调整字段需同步更新 offset。

## 结论
01 是 patch 必须回传的构建缺陷（rms_norm 依赖遗漏）。02 是 patch 必须回传的
测试缺陷（CANN 版本兼容 + 读回方式）。两者都不改算子数学。
