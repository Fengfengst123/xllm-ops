# A5 GammaAddRmsNorm validation prompt

你现在位于一台 Ascend A5/950 服务器的全新容器内。请完整验证
`GammaAddRmsNorm` 的 arch35 实现，包括干净构建、Host tiling UT、ATK
精度、ATK 动态 shape 性能，以及 msprof device-task 证据。不要只给方案，
持续执行到验证完成或出现无法自行解决的外部阻塞。

## 输入文件和代码来源

直接拉取包含算子、CANN 公共版本兼容修复和验证资产的 xllm-ops 分支：

```bash
set -euo pipefail
mkdir -p /workspace/artifacts
git clone --branch codex/gamma-add-rmsnorm-a5-validation --single-branch \
  https://github.com/Fengfengst123/xllm-ops.git /workspace/xllm-ops
cd /workspace/xllm-ops
git merge-base --is-ancestor 0ab5c2998cd5a30087a94dcb5211b2fe31b5741c HEAD
git status --short | tee /workspace/artifacts/xllm_ops_status.txt
test -z "$(git status --porcelain)"
```

拉取后必须验证以下事实：

```bash
cd /workspace/xllm-ops
test -f xllm_ops/gamma_add_rms_norm/op_kernel/arch35/gamma_add_rms_norm_regbase.h
test -f xllm_ops/gamma_add_rms_norm/op_kernel/arch35/gamma_add_rms_norm_regbase_split_d.h
test -f xllm_ops/gamma_add_rms_norm/op_host/gamma_add_rms_norm_error_log.h
! rg 'GetStorageShape\(\)\.ToString|\.ToString\(\)' \
  xllm_ops/gamma_add_rms_norm/op_host
rg 'Ops::Base::ToString' xllm_ops/gamma_add_rms_norm/op_host/gamma_add_rms_norm_tiling.cpp
rg 'op_common/op_host/util/math_util.h' \
  xllm_ops/gamma_add_rms_norm/op_host/gamma_add_rms_norm_tiling_arch35.cpp
rg 'aclnn_util.h' \
  xllm_ops/gamma_add_rms_norm/op_host/op_api/aclnn_gamma_add_rms_norm.h
rg '#include "op_api_def.h"' \
  xllm_ops/gamma_add_rms_norm/op_host/op_api/aclnn_gamma_add_rms_norm.cpp
```

不要把 `gert::Shape` 转成 `std::to_string(shape)`；正确兼容方式是
`Ops::Base::ToString(shape)`。

## 1. 环境门禁

自动找到有效的 CANN `set_env.sh`，优先使用 `latest`。记录但不要修改系统全局环境：

```bash
if test -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
elif test -f /usr/local/Ascend/ascend-toolkit/set_env.sh; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
  echo 'CANN set_env.sh not found' >&2
  exit 2
fi

npu-smi info | tee /workspace/artifacts/npu_smi_before.txt
env | sort | rg 'ASCEND|CANN|LD_LIBRARY_PATH|PYTHONPATH' \
  | tee /workspace/artifacts/ascend_env.txt
which ccec atk msprof || true
python3 -m pip show ATK torch torch-npu 2>&1 \
  | tee /workspace/artifacts/python_packages.txt
```

确认设备确实为 A5/Ascend 950 系列，并选择一张没有其他进程、AICore 空闲、HBM
没有异常占用的卡。设为 `A5_DEVICE`。如果所有卡均被占用，不要杀其他用户进程，
报告阻塞。

```bash
export A5_DEVICE=<空闲物理卡号>
export ASCEND_RT_VISIBLE_DEVICES=${A5_DEVICE}
```

## 2. 从零构建 xllm-ops Ascend 950 包

必须使用全新目录，禁止复用旧 build、output 或安装产物：

```bash
rm -rf /workspace/build_gamma_a5 /workspace/install_gamma_a5
mkdir -p /workspace/build_gamma_a5 /workspace/install_gamma_a5
cd /workspace/xllm-ops
bash build.sh \
  --build-dir=/workspace/build_gamma_a5 \
  --op-name=gamma_add_rms_norm \
  --compute-unit=ascend950 \
  2>&1 | tee /workspace/artifacts/build_gamma_a5.log
```

构建后检查：

```bash
find /workspace/build_gamma_a5 /workspace/xllm-ops/output -type f \
  \( -name '*.run' -o -name 'libop_host_aclnn.so' -o -name 'libcust_opapi.so' \
     -o -path '*ascend950*' \) 2>/dev/null \
  | sort | tee /workspace/artifacts/build_artifacts.txt
```

必须确认存在 Ascend 950 kernel 和 ACLNN host/op-api 产物。安装生成的 `.run`
包到 `/workspace/install_gamma_a5`；先查看安装器 `--help`，按它支持的非交互参数
安装，不要猜参数。安装后 source 它生成的 `set_env.bash`。记录准确路径：

```bash
find /workspace/install_gamma_a5 -type f \
  \( -name 'set_env.bash' -o -name 'libop_host_aclnn.so' \
     -o -name 'libcust_opapi.so' \) | sort \
  | tee /workspace/artifacts/install_artifacts.txt
```

设置 `GAMMA_OPAPI_LIB` 为实际包含
`aclnnGammaAddRmsNormGetWorkspaceSize`/`aclnnGammaAddRmsNorm` 的库，并校验符号：

```bash
export GAMMA_OPAPI_LIB=<实际自定义op-api库绝对路径>
test -f "$GAMMA_OPAPI_LIB"
nm -D "$GAMMA_OPAPI_LIB" | rg 'aclnnGammaAddRmsNorm(GetWorkspaceSize)?$'
```

## 3. 准备 ops-nn 验证仓

xllm-ops 传输包内有两个 ops-nn patch，必须应用它们；性能和 ATK executor
依赖这个仓库：

```bash
rm -rf /workspace/ops-nn
git clone https://gitcode.com/cann/ops-nn.git /workspace/ops-nn
cd /workspace/ops-nn
git fetch origin \
  refs/keep-around/a6268b3a09da05f11012a2f8ee714f8a951a4f54:refs/heads/gamma-validation-base
git checkout gamma-validation-base
zcat /workspace/xllm-ops/validation/arch35/ops-nn-patches/0001-feat-norm-add-GammaAddRmsNorm-custom-op.patch.gz \
  | git am
zcat /workspace/xllm-ops/validation/arch35/ops-nn-patches/0001-test-norm-cover-gamma-offset-on-arch35.patch.gz \
  | git am
git log -2 --oneline | tee /workspace/artifacts/ops_nn_patch_log.txt
```

预期 subject：

```text
test(norm): cover gamma offset on arch35
feat(norm): add GammaAddRmsNorm custom op
```

## 4. Host tiling UT

执行 key 1000（full-load gamma）和 key 2000（split-D gamma）的 Host UT：

```bash
cd /workspace/ops-nn
bash build.sh -u --ophost --ops=gamma_add_rms_norm --soc=ascend950 -j16 \
  2>&1 | tee /workspace/artifacts/gamma_arch35_host_ut.log
```

如果当前 ops-nn CLI 参数有变化，先执行 `bash build.sh --help`，只做等价调整。
必须给出执行 case 数、通过数、失败数，并证明两条 tiling key 都验证了
`addGammaOffset == 1`。

## 5. ATK 全量精度

先跑固定 240-case 矩阵，不能用一个固定 shape 冒充全量：

```bash
command -v atk
export ST_DIR=/workspace/ops-nn/norm/gamma_add_rms_norm/tests/st/aclnnGammaAddRmsNorm
test -f "${ST_DIR}/atk_aclnnGammaAddRmsNorm_full.json"

mkdir -p /workspace/atk_accuracy_a5
cd /workspace/atk_accuracy_a5
atk task \
  -c "${ST_DIR}/atk_aclnnGammaAddRmsNorm_full.json" \
  -n /workspace/xllm-ops/validation/arch35/atk_node_pyaclnn_cpu.yaml \
  -p "${ST_DIR}" \
  --task accuracy --single_process --max_task 1 --log info \
  2>&1 | tee /workspace/artifacts/atk_accuracy_a5.log
```

精度验收必须同时满足：

- 240/240 通过，0 失败；如果实际 patch 生成数量变化，报告真实总数且必须 0 失败。
- FP16、BF16、FP32 均执行。
- `addGammaOffset=true/false` 均执行。
- 使用 PyAclnn 自定义库和 CPU 双标杆，不能误加载系统同名算子。
- 比较三个输出：`y`、FP32 `rstd`、`xOut`。
- 保留 ATK Excel、完整日志和生成数据路径。

再从 YAML 重新生成一轮 broader matrix，验证 YAML 生成链路：

```bash
cd "${ST_DIR}"
rm -rf result/aclnnGammaAddRmsNorm
atk case -f aclnnGammaAddRmsNorm.yaml -p generate_gamma_add_rms_norm.py
python3 - <<'PY'
import json
p = 'result/aclnnGammaAddRmsNorm/json/all_aclnnGammaAddRmsNorm.json'
cases = json.load(open(p))
print('generated_cases=', len(cases))
assert len(cases) > 1
PY

atk task \
  -c result/aclnnGammaAddRmsNorm/json/all_aclnnGammaAddRmsNorm.json \
  -n /workspace/xllm-ops/validation/arch35/atk_node_pyaclnn_cpu.yaml \
  -p "${ST_DIR}" \
  --task accuracy --single_process --max_task 1 --log info \
  2>&1 | tee /workspace/artifacts/atk_accuracy_a5_generated.log
```

## 6. ATK 动态 shape 性能

必须使用分支里的 YAML 生成 shape。Python generator 只能建立输入依赖：

```text
x2.shape = x1.shape
gamma.shape = [x1.shape[-1]]
```

禁止在 generator 中写死完整 `[M,D]` shape，禁止只测一个 shape。

```bash
export PERF_DIR=/workspace/xllm-ops/validation/arch35/atk_dynamic
cd "$PERF_DIR"
rm -rf result cases plugins gamma_vs_add_full gamma_vs_old_chain_full
atk case -f gamma_add_rms_norm_perf.yaml \
  -p generate_gamma_add_rms_norm_dynamic_perf.py
python3 prepare_aligned_cases.py \
  result/gamma_add_rms_norm_perf/json/all_gamma_add_rms_norm_perf.json cases

python3 - <<'PY'
import json
cases = json.load(open('cases/shape_manifest.json'))
shapes = {tuple(x['x_shape']) for x in cases}
print('case_count=', len(cases), 'unique_shapes=', len(shapes))
assert len(cases) >= 30
assert len(shapes) == len(cases)
for x in cases:
    assert x['gamma_shape'] == [x['x_shape'][-1]]
PY

mkdir -p plugins
cp "${ST_DIR}/executor_aclnnGammaAddRmsNorm.py" plugins/
cp old_chain_perf_api.py plugins/
```

分别运行两个完全对齐的动态 shape 矩阵：

1. `GammaAddRmsNorm(true)` vs 裸 `AddRmsNorm`。
2. `GammaAddRmsNorm(true)` vs 等价旧链路 `Add(gamma,1)+AddRmsNorm`。

```bash
mkdir -p gamma_vs_add_full
(
  cd gamma_vs_add_full
  atk task -n ../perf_nodes.yaml \
    -c ../cases/gamma_vs_add_rms_norm_cases.json \
    --task performance_device -p ../plugins \
    --performance_data 100,20,10 -sp --save_data profile -l error \
    2>&1 | tee atk.log
  test "${PIPESTATUS[0]}" -eq 0
)

mkdir -p gamma_vs_old_chain_full
(
  cd gamma_vs_old_chain_full
  atk task -n ../perf_nodes.yaml \
    -c ../cases/gamma_vs_old_chain_cases.json \
    --task performance_device -p ../plugins \
    --performance_data 100,20,10 -sp --save_data profile -l error \
    2>&1 | tee atk.log
  test "${PIPESTATUS[0]}" -eq 0
)

python3 summarize_atk_perf.py \
  2>&1 | tee /workspace/artifacts/atk_dynamic_perf_summary.log
cp RESULTS.md performance_results.csv /workspace/artifacts/
```

不要直接复用脚本里任何来自 910B3 的历史结论。A5 必须按本机新生成的
`op_summary.csv` 重新计算。报告全矩阵，并单独报告：

- `D >= 1024` 模型相关子集。
- `M <= 31 且 D >= 1024` decode-like 子集。
- A5 key 1000 和 key 2000 子集。

性能口径：

```text
old chain = mean(Add task duration) + mean(AddRmsNorm task duration)
new chain = mean(GammaAddRmsNorm task duration)
```

必须同时给出 ATK Excel 自带的 verdict/平均性能比，以及逐 case
`op_summary.csv` 计算的算术平均、几何平均、median、通过 shape 数。

## 7. Direct smoke 与 msprof 复核

编译分支自带 harness：

```bash
cd /workspace/xllm-ops
CANN_HOME=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
g++ -std=c++17 -O2 validation/arch35/direct_aclnn_bench.cpp \
  -I"${CANN_HOME}/include" -L"${CANN_HOME}/lib64" \
  -Wl,-rpath,"${CANN_HOME}/lib64" -lascendcl -lnnopbase -ldl \
  -o /workspace/direct_aclnn_bench

export ADD_OPAPI_LIB=$(find /usr/local/Ascend -type f -name libopapi_nn.so | head -1)
export ADDS_OPAPI_LIB=$(find /usr/local/Ascend -type f -name libopapi_math.so | head -1)
test -f "$ADD_OPAPI_LIB" && test -f "$ADDS_OPAPI_LIB"
```

至少覆盖 BF16/FP16/FP32、key 1000 和 key 2000：

```bash
for shape in '4 2048' '4 8192'; do
  set -- $shape
  X_ROWS=$1 X_COLS=$2 WARMUP=20 ITERS=100 RUN_SMOKE=1 \
    /workspace/direct_aclnn_bench \
    2>&1 | tee "/workspace/artifacts/direct_smoke_${1}x${2}.log"
done
```

如果 8192 没命中 key 2000，依次尝试 16384、32768，直到 runtime 证据确认。

对以下 BF16 shape 每个跑 5 轮：

```bash
mkdir -p /workspace/artifacts/direct_perf
for shape in '1 2048' '4 2048' '31 2048' \
             '1 5120' '4 5120' '31 5120' '4 8192'; do
  set -- $shape
  for round in 1 2 3 4 5; do
    CASE_FILTER=bf16 X_ROWS=$1 X_COLS=$2 \
      WARMUP=30 ITERS=200 RUN_SMOKE=0 \
      /workspace/direct_aclnn_bench \
      2>&1 | tee "/workspace/artifacts/direct_perf/${1}x${2}_r${round}.log"
  done
done
```

event timing 只能作 smoke，不得作为最终性能结论。最终必须用 msprof：

```bash
mkdir -p /workspace/artifacts/msprof
for shape in '1 2048' '4 2048' '31 2048' \
             '1 5120' '4 5120' '31 5120' '4 8192'; do
  set -- $shape
  CASE_FILTER=bf16 X_ROWS=$1 X_COLS=$2 WARMUP=30 ITERS=200 RUN_SMOKE=0 \
  msprof --output="/workspace/artifacts/msprof/${1}x${2}" \
    --application=/workspace/direct_aclnn_bench \
    2>&1 | tee "/workspace/artifacts/msprof_${1}x${2}.log"
done
```

从每个 `op_summary.csv` 汇总 `Add`、`AddRmsNorm`、`GammaAddRmsNorm` 的
device `Task Duration(us)`。不得拿 API/event 时间与 device task 时间混比。

## 8. 验收和最终汇报

最终报告必须包含：

1. A5 型号、卡号、CANN/Driver/ATK/torch-npu 版本。
2. xllm-ops 分支 commit、ops-nn 两个 patch subject。
3. 从零构建命令、退出码、Ascend 950 kernel/op-api/package 路径。
4. Host UT：key 1000/2000 case 数和通过情况。
5. ATK 精度：总数/通过/失败、dtype、offset 模式、Excel 绝对路径。
6. ATK 动态性能：全部生成 shape、ATK verdict、逐 shape device 数据。
7. msprof 表格：shape、tiling key、Add、AddRmsNorm、旧链路总耗时、
   GammaAddRmsNorm、加速比、降幅、5 轮波动。
8. 明确结论：
   - `GammaAddRmsNorm(false)` 是否与裸 AddRmsNorm 精度一致。
   - `GammaAddRmsNorm(true)` 是否与旧双算子链路精度一致。
   - 相对裸 AddRmsNorm 是否在正常噪声范围。
   - 相对旧链路是否稳定获益。
9. 所有日志和报告必须保存在 `/workspace/artifacts`，列出绝对路径。

失败时不要掩盖或跳过：保留完整命令、错误、退出码和最小必要修复 diff。
不要修改算子数学逻辑来让测试通过；任何兼容修复先说明根因和影响范围。
