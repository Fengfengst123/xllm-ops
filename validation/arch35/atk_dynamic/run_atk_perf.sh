#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
: "${INSTALL_ROOT:?set INSTALL_ROOT to the custom-op installation root}"
: "${ST_DIR:?set ST_DIR to the ops-nn GammaAddRmsNorm ST directory}"
DEVICE=${DEVICE:-0}

OPP=${OPP:-"$INSTALL_ROOT/vendors/custom_xllm_math"}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}

source "$CANN_ENV"
source "$OPP/bin/set_env.bash"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export GAMMA_OPAPI_LIB=${GAMMA_OPAPI_LIB:-"$OPP/op_api/lib/libcust_opapi.so"}

test -f "$GAMMA_OPAPI_LIB"
test -f "$ST_DIR/executor_aclnnGammaAddRmsNorm.py"

cd "$ROOT"
rm -rf result cases plugins gamma_vs_add_full gamma_vs_old_chain_full
atk case -f gamma_add_rms_norm_perf.yaml \
  -p generate_gamma_add_rms_norm_dynamic_perf.py
python3 prepare_aligned_cases.py \
  result/gamma_add_rms_norm_perf/json/all_gamma_add_rms_norm_perf.json cases

mkdir -p plugins
cp "$ST_DIR/executor_aclnnGammaAddRmsNorm.py" plugins/
cp old_chain_perf_api.py plugins/

for comparison in add_rms_norm old_chain; do
  run_dir="gamma_vs_${comparison}_full"
  case_file="cases/gamma_vs_${comparison}_cases.json"
  mkdir -p "$run_dir"
  (
    cd "$run_dir"
    atk task -n ../perf_nodes.yaml -c "../$case_file" \
      --task performance_device -p ../plugins \
      --performance_data 100,20,10 -sp --save_data profile -l error \
      2>&1 | tee atk.log
  )
done

python3 summarize_atk_perf.py
