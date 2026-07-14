# xllm-ops 构建修复（ascend950/A5 支持）

> 仓库: Fengfengst123/xllm-ops, 分支 codex/gamma-add-rmsnorm-a5-validation
> HEAD = eb016ba5f4bdd169f798a7d8e40e0a03e582abb2
> CANN: 9.1.0-beta.1, 设备: Ascend950PR (NpuArch 3510, soc Ascend950PR_9599)

这些修复让 xllm-ops 构建系统支持 A5/Ascend950。当前分支此前只能构建
ascend310p/ascend910b/ascend910_93，ascend950 会落到 else 分支直接 skip。

## 01_build_aclnn_ascend950_branch.diff
- 文件: xllm_ops/build_aclnn.sh
- 根因: build_aclnn.sh 只有 ascend310/910b/910_93 三个 SOC 分支，ascend950
  命中 else -> 打印 "no custom ACLNN ops configured" 并 exit 0，整个构建被跳过，
  不产出 .run 包。
- 修复: 新增 `elif [[ "$SOC_VERSION" =~ ^ascend950 ]]` 分支，CUSTOM_OPS 含
  gamma_add_rms_norm，SOC_ARG=ascend950（与 op 的 AddConfig("ascend950") 一致）。
- 影响: 仅 build 派发；不改任何算子数学/tiling/kernel。
- 回传建议: **应回传 xllm-ops 分支**。这是 A5 构建的必要门控，缺失则任何 A5
  环境都无法用 xllm-ops build.sh 构建 ascend950 包。

## 02_build_sh_soc_naming.diff
- 文件: xllm_ops/build.sh
- 根因: SUPPORT_COMPUTE_UNIT_SHORT 用的是陈旧名 `ascend910_95`，且
  process_soc_input 里 `sed 's/ascend950/ascend910_95/g'` 把 ascend950 强制
  改名。但 CANN 9.1.0-beta.1 和 ops-nn 仓都用 `ascend950`（op 的
  AddConfig("ascend950") 也用此名），导致 ASCEND_COMPUTE_UNIT=ascend910_95 与
  opbuild 生成的 aic-ascend950-ops-info.ini 名字不匹配，cp 失败。
- 修复: SUPPORT_COMPUTE_UNIT_SHORT 改回 `ascend950`；删除
  ascend950->ascend910_95 的 sed 翻译。
- 影响: 仅 build SOC 名；不改算子。
- 回传建议: **应回传 xllm-ops 分支**。与 01 配套，A5 构建必需。

## 03_cmakelists_ascend950.diff
- 文件: xllm_ops/CMakeLists.txt
- 根因: ASCEND_ALL_COMPUTE_UNIT / SOC_VERSION_LIST 仍用 `ascend910_95`，
  CMakeLists 的 SOC 查表（line 47-60）对 ASCEND_COMPUTE_UNIT=ascend950 命中
  else -> "unsupported chip type" -> 走 build_empty_package -> CPack 找不到
  packaging 文件失败。
- 修复: ascend910_95 -> ascend950（3 处：ALL 列表、SOC_VERSION_LIST、
  BUILD_WITH_3_8_PACKAGE 判定）。
- 影响: 仅 CMake SOC 派发 + arch35 目录映射（ascend950 -> arch35，保持不变）。
- 回传建议: **应回传 xllm-ops 分支**。与 01/02 配套。

## 04_const_var_ascend950pr_9599.diff
- 文件: cmake/scripts/util/const_var.py
- 根因: SOC_MAP_EXT 没有 ascend950 项，conv_soc_ver 返回 None 后回退
  `self.soc.capitalize()` = "Ascend950"。但 CANN 9.1.0-beta.1 的 platform
  没有泛化 Ascend950，只有具体型号 Ascend950PR_9599 / Ascend950PR_9579 /
  Ascend950DT_* 等。opc --soc_version=Ascend950 报
  "get platform info failed" (EB9000)。
- 修复: SOC_MAP_EXT 增加 `"ascend950": "Ascend950PR_9599"`（与 ops-nn
  func.cmake map_compute_unit 的 ascend950->ascend950pr_9599 一致；这是 A5
  的 canonical 编译目标型号）。
- 影响: 仅 kernel 编译时的 soc_version 字符串；不改算子。
- 回传建议: **应回传 xllm-ops 分支**。注意：若 xllm-ops 要支持多种 A5 子型号
  （如 Ascend950DT_*），需进一步泛化（按 npu-smi 实际型号选择），当前固定
  9599 是最小可用值，与 ops-nn 一致。

## 05_vendored_kernel_deps.diff
- 文件: xllm_ops/gamma_add_rms_norm/{op_kernel/inc/platform.h, rms_norm/*, norm_common/*}
- 根因: gamma 的 arch35 kernel 头 include `../inc/platform.h`、
  `../../rms_norm/...`、`../../norm_common/...`。这些在 ops-nn 由
  common/inc/op_kernel/ 共享机制 + sibling op 布局提供，但 xllm-ops 没有这套
  机制，也没有 rms_norm/norm_common op 目录。
- 修复（临时）: 从 CANN ascendc/inc/platform.h 和 ops-nn 的 rms_norm/norm_common
  逐字拷贝 5 个 arch35 kernel 头到 gamma op 下（相对路径与 include 对齐）。
- 注意: 这是 xllm-ops 构建上下文的 **workaround**。即便如此，xllm-ops 的
  kernel 编译仍因 `#include "kernel_utils.h"` 解析到 CANN basic_api 版本（无
  namespace ops）而失败——这是 xllm-ops 与 ops-nn 构建上下文的深层差异。
  xllm-ops 的 op_host/aclnn 能构建成功并产出正确符号；kernel 二进制最终用
  ops-nn（op 的原生构建系统）构建。
- 回传建议: **不回传**。真正修复应是 ① 给 xllm-ops 增加类似 ops-nn 的
  common/inc/op_kernel 共享机制 + ascendc/inc include 路径，或 ② 文档说明 A5
  kernel 必须用 ops-nn 构建。vendoring 会与上游 rms_norm/norm_common 漂移。
  （配套的真正修复见 ops-nn/01：给 gamma op 的 DEPENDENCIES 补 rms_norm。）

## 结论
01-04 是 xllm-ops A5 构建门控的必要修复，**应回传**。05 是临时 workaround，
不应回传；A5 kernel 构建应走 ops-nn（见 ops-nn/01 的 DEPENDENCIES 修复）。
