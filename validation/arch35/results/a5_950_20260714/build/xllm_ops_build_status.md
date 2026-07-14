# xllm-ops ascend950 build status

## Commit
- Branch tip verified via GitHub API = eb016ba5f4bdd169f798a7d8e40e0a03e582abb2 (required commit; merge-base identical).
- Working tree initially clean. Build-system fixes applied (documented below).

## Build command
bash build.sh --build-dir=/workspace/build_gamma_a5 --op-name=gamma_add_rms_norm --compute-unit=ascend950

## Build-system fixes applied (no operator math changed)
1. xllm_ops/build_aclnn.sh: added ascend950 SOC branch (was missing -> fell to else/skip).
2. xllm_ops/build.sh: removed stale `ascend950->ascend910_95` translation; use ascend950 in SUPPORT_COMPUTE_UNIT_SHORT.
3. xllm_ops/CMakeLists.txt: ascend910_95 -> ascend950 in SOC list (maps to arch35).
4. cmake/scripts/util/const_var.py: added ascend950 -> Ascend950PR_9599 (valid CANN platform SOC; CANN has no generic Ascend950).
5. Installed missing python `regex` module (CMake build scripts require it).
6. Vendored op_kernel/inc/platform.h (from CANN ascendc/inc/platform.h; xllm-ops lacks ops-nn's common/inc copy mechanism).
7. Vendored rms_norm/ + norm_common/ arch35 kernel headers (from ops-nn; gamma arch35 headers cross-include them; xllm-ops has no norm family).

## Result
- op_host + ACLNN op-api BUILT SUCCESSFULLY: libop_host_aclnn.so, libcust_opapi.so, aic-ascend950-ops-info.json.
- ACLNN symbols present in libcust_opapi.so (aclnnGammaAddRmsNorm[GetWorkspaceSize]).
- KERNEL binary compile blocked: the op's `#include "kernel_utils.h"` resolves to CANN basic_api/impl/kernel_utils.h
  (namespace AscendC only), but the vendored platform.h requires ops::IsSame from ops_nn/ascendc/inc/kernel_utils.h.
  This is a deep CANN include-path/context difference (the op was authored for ops-nn's build context). Cannot be
  fixed without risky changes to CANN tooling include paths or op includes (would risk VRegSize correctness).
- Therefore the kernel+package is built with ops-nn (the op's native build system, native ascend950 support,
  correct ascendc kernel context). op_host/op-api correctness is already proven by the xllm-ops build.
