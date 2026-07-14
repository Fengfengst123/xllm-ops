# xllm-ops ascend950 build status (v2 - corrected)

## Commit
- Branch: codex/gamma-add-rmsnorm-a5-validation
- HEAD = eb016ba5f4bdd169f798a7d8e40e0a03e582abb2 (required baseline commit)
- Working tree initially clean. Build-system fixes applied on results branch.

## Build command
```bash
bash build.sh --build-dir=/workspace/build_gamma_a5 \
  --op-name=gamma_add_rms_norm --compute-unit=ascend950
```

## Build result: SUCCESS (exit code 0)

xllm-ops now **independently builds a complete ascend950 package** including:
- arch35 kernel binary (3 .o files: fp16/bf16/fp32 tiling key variants)
- op_host + ACLNN op-api (libcust_opapi.so with aclnnGammaAddRmsNorm symbols)
- .run installer package

### Kernel artifacts
```
/workspace/build_gamma_a5/binary/ascend950/bin/gamma_add_rms_norm_apt/
  GammaAddRmsNorm_c549aa7cfbf8ccffe3b5fbc767e383f6.o
  GammaAddRmsNorm_fd94913b5f28c2606574554b42dbd654.o
  GammaAddRmsNorm_1106d4daca8765f5bb4811fcb0614277.o
```

### .run package
```
/workspace/build_gamma_a5/cann-ops-xllm-custom_linux-x86_64.run
```

### SHA256
(see build/sha256_manifest.txt in results directory)

### ACLNN symbols (verified)
```
nm -D libcust_opapi.so | grep aclnnGammaAddRmsNorm
  T aclnnGammaAddRmsNorm
  T aclnnGammaAddRmsNormGetWorkspaceSize
```

### Installation
Installed to /usr/local/Ascend/cann-9.1.0-beta.1/opp/vendors/custom_xllm_math/
- config.ini written (load_priority=custom_xllm_math,custom_xllm_math_nn)
- set_env.bash generated

## Build-system fixes applied (no operator math changed)

### ascend950 SOC support (must backport)
1. **build_aclnn.sh**: added `ascend950` SOC branch (was missing → fell to else/skip)
2. **build.sh**: `ascend910_95` → `ascend950` in SUPPORT_COMPUTE_UNIT_SHORT; removed stale `ascend950→ascend910_95` sed translation
3. **CMakeLists.txt**: `ascend910_95` → `ascend950` in SOC list (maps to arch35)
4. **const_var.py**: added `ascend950` → `Ascend950PR_9599` SOC map (CANN 9.1 has no generic Ascend950)

### Kernel include fix (must backport)
5. **platform.h**: removed dead-code `IsSupportAtomicAddTypeSIMD()` (depended on `ops::IsSame` from `kernel_utils.h` which resolves to wrong header in xllm-ops context); dropped `kernel_utils.h` include. Gamma kernel only needs `GetUbBlockSize()`/`GetVRegSize()`.
6. **rms_norm_regbase_common.h**: dropped unused `kernel_utils.h` include
7. **gamma_add_rms_norm_regbase_common.h**: dropped unused `kernel_utils.h` include

### Link fix (must backport)
8. **symbol.cmake**: cust_opmaster links `libops_base.so` (IMPORTED target opsbase not visible in function scope on CANN 9.1; fixes `undefined symbol: Ops::Base::ToString`)
9. **custom_build.cmake**: cust_opmaster + cust_proto link `libops_base.so` (same scope issue)
10. **stub/op_tiling/CMakeLists.txt**: optiling links `libops_base.so`

### Vendored kernel deps (genuine cross-op dependencies)
11. `op_kernel/inc/platform.h`: platform adapter (GetUbBlockSize/GetVRegSize)
12. `rms_norm/`: KernelRmsNormBase + arch35 regbase common (ComputeRstd/ComputeY etc.)
13. `norm_common/`: reduce_common_regbase

These are genuine cross-op kernel dependencies that gamma's arch35 headers include via `../../rms_norm/` and `../../norm_common/`. In ops-nn these are provided via a shared `common/inc/op_kernel/` mechanism; xllm-ops lacks this mechanism, so the headers are vendored.

## Conclusion
xllm-ops **can independently build and run** the GammaAddRmsNorm arch35 kernel on A5. The package was verified with direct_aclnn_bench (3 dtypes PASS, max_abs=0) and ATK (6 dtype×offset smoke all Pass).
