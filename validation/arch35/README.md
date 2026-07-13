# GammaAddRmsNorm Ascend 950 validation

This directory contains the assets needed to validate the arch35 implementation
on an isolated Ascend 950 container.

## Repositories

The xllm-ops branch already contains the operator implementation:

```bash
git clone --branch codex/gamma-add-rmsnorm-arch35 --single-branch \
  https://github.com/Fengfengst123/xllm-ops.git /workspace/xllm-ops
cd /workspace/xllm-ops
git rev-parse HEAD
# Expected: 0ab5c2998cd5a30087a94dcb5211b2fe31b5741c
```

The ops-nn validation tree is reconstructed from the public parent commit and
the two patches in this directory:

```bash
git clone https://gitcode.com/cann/ops-nn.git /workspace/ops-nn
cd /workspace/ops-nn
git fetch origin \
  refs/keep-around/a6268b3a09da05f11012a2f8ee714f8a951a4f54:refs/heads/gamma-validation-base
git checkout gamma-validation-base
zcat /workspace/xllm-ops/validation/arch35/ops-nn-patches/0001-feat-norm-add-GammaAddRmsNorm-custom-op.patch.gz \
  | git am
zcat /workspace/xllm-ops/validation/arch35/ops-nn-patches/0001-test-norm-cover-gamma-offset-on-arch35.patch.gz \
  | git am
git log -2 --oneline
```

Expected commits after applying the patches:

```text
5292c024da test(norm): cover gamma offset on arch35
b3a3c8170b feat(norm): add GammaAddRmsNorm custom op
```

## Build xllm-ops

Load the CANN environment first and verify that its compiler accepts
`ascend950`:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
cd /workspace/xllm-ops
bash build.sh --build-dir=/workspace/build_gamma_add_rms_norm_950 \
  --op-name=gamma_add_rms_norm --compute-unit=ascend950 \
  2>&1 | tee /workspace/build_gamma_add_rms_norm_950.log
```

The build must produce the host ACLNN library, arch35 kernel artifacts and
Ascend 950 binary configuration. Locate them rather than assuming a package
layout:

```bash
find /workspace/build_gamma_add_rms_norm_950 /workspace/xllm-ops/output \
  -type f \( -name 'libop_host_aclnn.so' -o -name '*.run' \
  -o -path '*ascend950*' \) 2>/dev/null | sort
```

Install the generated custom-op package using its `install.sh` or `.run`
installer, then source the generated environment script. Record the exact
installation path. Do not run ATK until a one-case ACLNN call is confirmed to
load the custom GammaAddRmsNorm library and its Ascend 950 kernel.

## Host tiling UT

The ops-nn validation patch adds assertions for both arch35 tiling paths:

- key 1000: full-load gamma, `addGammaOffset == 1`
- key 2000: split-D gamma, `addGammaOffset == 1`

Run the focused host UT with the repository's current build syntax:

```bash
cd /workspace/ops-nn
bash build.sh -u --ophost --ops=gamma_add_rms_norm --soc=ascend950 -j16 \
  2>&1 | tee /workspace/gamma_add_rms_norm_arch35_host_ut.log
```

## ATK accuracy

The committed fixed matrix contains 240 cases: 120 with gamma offset enabled
and 120 disabled. It covers FP16, BF16 and FP32 and is preferred for a
reproducible first full run.

```bash
export ST_DIR=/workspace/ops-nn/norm/gamma_add_rms_norm/tests/st/aclnnGammaAddRmsNorm
export GAMMA_OPAPI_LIB=$(find /workspace/build_gamma_add_rms_norm_950 \
  /workspace/xllm-ops/output -type f -name libop_host_aclnn.so \
  2>/dev/null | head -1)
test -n "${GAMMA_OPAPI_LIB}" && test -f "${GAMMA_OPAPI_LIB}"

atk task \
  -c "${ST_DIR}/atk_aclnnGammaAddRmsNorm_full.json" \
  -n /workspace/xllm-ops/validation/arch35/atk_node_pyaclnn_cpu.yaml \
  -p "${ST_DIR}" \
  --task accuracy --single_process --max_task 1 --log info \
  2>&1 | tee /workspace/atk_gamma_add_rms_norm_950_full.log
```

The `pyaclnn` node calls the custom ACLNN implementation. The CPU node uses
`executor_aclnnGammaAddRmsNorm.py` as the double benchmark. The executor checks
all three outputs: `y`, FP32 `rstd`, and `xOut`.

To regenerate a broader matrix from YAML instead of using the fixed JSON:

```bash
cd "${ST_DIR}"
rm -rf result/aclnnGammaAddRmsNorm
atk case -f aclnnGammaAddRmsNorm.yaml -p generate_gamma_add_rms_norm.py
python3 - <<'PY'
import json
p = 'result/aclnnGammaAddRmsNorm/json/all_aclnnGammaAddRmsNorm.json'
x = json.load(open(p))
print('generated_cases=', len(x))
PY

atk task \
  -c result/aclnnGammaAddRmsNorm/json/all_aclnnGammaAddRmsNorm.json \
  -n /workspace/xllm-ops/validation/arch35/atk_node_pyaclnn_cpu.yaml \
  -p "${ST_DIR}" \
  --task accuracy --single_process --max_task 1 --log info \
  2>&1 | tee /workspace/atk_gamma_add_rms_norm_950_generated.log
```

Before accepting the result, verify from the report and log:

1. All cases actually used `pyaclnn` and CPU nodes.
2. Both `addGammaOffset=true` and `false` ran.
3. FP16, BF16 and FP32 ran.
4. Both key 1000 and key 2000 appear in tiling/kernel evidence.
5. The report has zero failed cases. Report the exact total/pass/fail counts.

If `GAMMA_OPAPI_LIB` is ignored or the symbol cannot be loaded, stop and fix
the custom library/package path. A run against a system operator is not valid.

## Direct accuracy and performance comparison

Compile the supplied harness:

```bash
cd /workspace/xllm-ops
CANN_HOME=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
g++ -std=c++17 -O2 validation/arch35/direct_aclnn_bench.cpp \
  -I"${CANN_HOME}/include" -L"${CANN_HOME}/lib64" \
  -Wl,-rpath,"${CANN_HOME}/lib64" -lascendcl -lnnopbase -ldl \
  -o /workspace/direct_aclnn_bench

export ADD_OPAPI_LIB=$(find /usr/local/Ascend -type f -name libopapi_nn.so | head -1)
export ADDS_OPAPI_LIB=$(find /usr/local/Ascend -type f -name libopapi_math.so | head -1)
export GAMMA_OPAPI_LIB
```

The smoke section verifies:

- `GammaAddRmsNorm(true)` equals `Adds(gamma, 1) + AddRmsNorm`
- `GammaAddRmsNorm(false)` equals `AddRmsNorm`
- `y`, `rstd`, and `xOut` all match

Run smoke coverage for both expected arch35 paths and all dtypes:

```bash
for shape in '4 2048' '4 8192'; do
  set -- ${shape}
  X_ROWS=$1 X_COLS=$2 WARMUP=20 ITERS=100 RUN_SMOKE=1 \
    /workspace/direct_aclnn_bench | tee "/workspace/smoke_${1}x${2}.log"
done
```

Confirm the actual tiling keys from runtime logs. If 8192 does not select key
2000, increase the hidden size to 16384 or 32768 and rerun.

Run five performance rounds for each required shape:

```bash
mkdir -p /workspace/perf950
for shape in '1 2048' '4 2048' '31 2048' \
             '1 5120' '4 5120' '31 5120' '4 8192'; do
  set -- ${shape}; rows=$1; cols=$2
  for round in 1 2 3 4 5; do
    CASE_FILTER=bf16 X_ROWS=${rows} X_COLS=${cols} \
      WARMUP=30 ITERS=200 RUN_SMOKE=0 \
      /workspace/direct_aclnn_bench \
      | tee "/workspace/perf950/${rows}x${cols}_r${round}.log"
  done
done
```

Harness event timing is useful only as a smoke signal. The final performance
conclusion must use msprof device task duration:

```bash
mkdir -p /workspace/msprof950
CASE_FILTER=bf16 X_ROWS=4 X_COLS=2048 WARMUP=30 ITERS=200 RUN_SMOKE=0 \
msprof --output=/workspace/msprof950/4x2048 \
  --application=/workspace/direct_aclnn_bench
```

Repeat msprof for each reporting shape. Locate `op_summary.csv` and aggregate:

- `Add`/`Adds` on gamma `[hidden]`
- system `AddRmsNorm` on `[rows, hidden]`
- custom `GammaAddRmsNorm` on `[rows, hidden]`

Report device-task average and total duration. Compare:

```text
old chain = gamma Add + AddRmsNorm
new chain = GammaAddRmsNorm(addGammaOffset=true)
```

Provide a table with dtype, shape, tiling key, Add latency, AddRmsNorm latency,
old total, GammaAddRmsNorm latency, reduction, and five-run variation.

## Acceptance criteria

- Ascend 950 package and arch35 kernels build successfully.
- Host tiling UT verifies key 1000 and key 2000 attribute propagation.
- Fixed 240-case ATK matrix has zero failures.
- Both gamma-offset modes and all supported dtypes run on the real NPU.
- Direct smoke comparison passes for `y`, `rstd`, and `xOut`.
- The new trace has no independent gamma Add in the fused path.
- GammaAddRmsNorm is no slower than AddRmsNorm beyond normal noise.
- GammaAddRmsNorm is faster than the old Add + AddRmsNorm chain.

Preserve all build logs, ATK reports, direct harness logs, msprof directories,
and `op_summary.csv` files under `/workspace` and report their absolute paths.
