# shellcheck shell=bash
# One archived C48 configuration; only the PD/offload stack changes.

prepare_mooncake_c48_runtime() {
  {
    hostname
    lscpu
    timeout 20s rocm-smi --showproductname --showclocks --showpower --showmaxpower || true
    timeout 20s ibv_devinfo -l || true
  } > "${RUNTIME_LOG_DIR}/hardware-rank-${NODE_RANK}.txt" 2>&1
  local wheel="${ATOMESH_SCRIPT_DIR}/mooncake-dist/mooncake_transfer_engine_rocm-0.3.13-cp312-cp312-manylinux_2_35_x86_64.whl"
  printf '%s  %s\n' 7dfe1f9acec16843868dde28dcb1d3903aafeaf8b92a657239055581a8c44df6 "${wheel}" | sha256sum -c -
  uv venv --system-site-packages --python "$(command -v python3)" /tmp/mooncake-c48-venv
  uv pip install --python /tmp/mooncake-c48-venv/bin/python --no-deps "${wheel}"
  export PATH="/tmp/mooncake-c48-venv/bin:${PATH}"
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  /tmp/mooncake-c48-venv/bin/python "${ATOMESH_SCRIPT_DIR}/pd_mooncake_c48_runtime.py" \
    "${RUNTIME_LOG_DIR}/mooncake-runtime-rank-${NODE_RANK}.json"
  export PYTHONPATH="/tmp/mooncake-c48-site:${PYTHONPATH:-}"
  export PYTHONDONTWRITEBYTECODE=1
  cc -shared -fPIC -O2 -Wall -Wextra -Werror -I/opt/rocm/include \
    "${ATOMESH_SCRIPT_DIR}/mooncake-c48/hsa_error_trace.c" -ldl -pthread \
    -o /tmp/mooncake-c48-hsa-error-trace.so
  sha256sum /tmp/mooncake-c48-hsa-error-trace.so \
    > "${RUNTIME_LOG_DIR}/hsa-error-trace-rank-${NODE_RANK}.sha256"
  /tmp/mooncake-c48-venv/bin/python "${ATOMESH_SCRIPT_DIR}/mooncake-dist/verify_rocm_wheel.py" \
    > "${RUNTIME_LOG_DIR}/mooncake-package-rank-${NODE_RANK}.log" 2>&1
  /tmp/mooncake-c48-venv/bin/python - <<'PY'
import mooncake.engine
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import MooncakeConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
assert mooncake.engine.SUPPORT_HIP
print('Mooncake ROCm imports passed:', MooncakeConnector.__name__, MooncakeStoreConnector.__name__)
PY
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=$((18998 + ATOMESH_SERVICE_PORT_OFFSET))
  export MOONCAKE_MASTER_PORT=$((19051 + ATOMESH_SERVICE_PORT_OFFSET))
  # Separate caches preserve the normal service's cold-start compilation path.
  local device
  for device in 0 1 2 3 4 5 6 7; do
    MOONCAKE_KERNEL_EVIDENCE_DIR="${RUNTIME_LOG_DIR}/kernel-preflight-rank-${NODE_RANK}" \
    TRITON_CACHE_DIR="/tmp/mooncake-kernel-preflight-${NODE_RANK}-${device}" \
      /tmp/mooncake-c48-venv/bin/python -m vllm._mooncake_kernel_evidence "${device}"
  done
}

start_mooncake_c48_store() {
  export MOONCAKE_CONFIG_PATH="${RUNTIME_LOG_DIR}/mooncake-store.json"
  /tmp/mooncake-c48-venv/bin/python - "${ATOMESH_SCRIPT_DIR}/../../benchmark/mooncake-c48.json" <<'PY'
import json
import os
from pathlib import Path
import sys
config = json.loads(Path(sys.argv[1]).read_text())['mooncake_store']
assert config['global_segment_size'] * 8 == 1799000000000
config['master_server_address'] = '127.0.0.1:' + os.environ['MOONCAKE_MASTER_PORT']
Path(os.environ['MOONCAKE_CONFIG_PATH']).write_text(json.dumps(config, indent=2) + '\n')
PY
  local master
  master="$(/tmp/mooncake-c48-venv/bin/python -c 'from pathlib import Path; import mooncake; print(Path(mooncake.__file__).parent / "mooncake_master")')"
  start_logged_process mooncake_master_pid "${RUNTIME_LOG_DIR}/mooncake-master.log" \
    "${master}" --port "${MOONCAKE_MASTER_PORT}"
  /tmp/mooncake-c48-venv/bin/python - "${MOONCAKE_MASTER_PORT}" <<'PY'
import socket
import sys
import time
for attempt in range(60):
    try:
        with socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Mooncake master startup timed out')
PY
}

start_vllm_server() {
  local role="$1" log_name="$2" port="$3"
  if [[ "${role}" == "prefill" ]]; then
    start_mooncake_c48_store
  fi
  local -a cache_env=()
  build_server_cache_env "${role}" "${port}" cache_env
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" \
    env "${cache_env[@]}" MOONCAKE_KERNEL_EVIDENCE_DIR="${RUNTIME_LOG_DIR}/kernel-warmup-${role}" \
    LD_PRELOAD="/tmp/mooncake-c48-hsa-error-trace.so${LD_PRELOAD:+:${LD_PRELOAD}}" \
    /tmp/mooncake-c48-venv/bin/python "${ATOMESH_SCRIPT_DIR}/pd_mooncake_c48_launch.py" \
    --config "${ATOMESH_SCRIPT_DIR}/../../benchmark/mooncake-c48.json" \
    --role "${role}" --model "${MODEL_PATH}" --host "${host_ip}" --port "${port}" \
    --output "${RUNTIME_LOG_DIR}/${log_name}.launch.json" --execute
}

start_router() {
  start_logged_process router_pid "${RUNTIME_LOG_DIR}/mooncake-proxy.log" \
    /tmp/mooncake-c48-venv/bin/python "${ATOMESH_SCRIPT_DIR}/pd_mooncake_c48_proxy.py" \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --prefill "http://${prefill_ips[0]}:${prefill_ports[0]}" "${VLLM_MOONCAKE_BOOTSTRAP_PORT}" \
    --decode "http://${decode_ips[0]}:${decode_ports[0]}"
}

ensure_aiperf() {
  git clone https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_DIR}"
  git -C "${AIPERF_DIR}" fetch origin "${AIPERF_COMMIT}"
  git -C "${AIPERF_DIR}" checkout --detach "${AIPERF_COMMIT}"
  # Match InferenceX infx/bench/agentic/venv.py, independently of server Python.
  uv venv --python 3.11 "${AIPERF_VENV}"
  UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 uv pip install \
    --python "${AIPERF_VENV}/bin/python" -e "${AIPERF_DIR}" 'datasets>=4.7.0'
  env -u PYTHONPATH "${AIPERF_VENV}/bin/aiperf" --version
}

run_aiperf_agentic_benchmark() {
  ensure_aiperf
  local out_dir="${RUN_DIR}/benchmark_results/mooncake-c48"
  mkdir -p "${out_dir}"
  git -C "${AIPERF_DIR}" rev-parse HEAD > "${out_dir}/aiperf-commit.txt"
  uv pip freeze --python "${AIPERF_VENV}/bin/python" > "${out_dir}/client-packages.txt"
  "${AIPERF_VENV}/bin/python" --version > "${out_dir}/client-python.txt"
  snapshot_eval_metrics "${out_dir}" before
  env -u PYTHONPATH AIPERF_EXPERIMENTAL_FAST=0 \
    AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=1 \
    AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false \
    AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=3600 \
    AIPERF_DATASET_MMAP_CACHE_DIR=/tmp/mooncake-c48-aiperf-mmap-cache \
    "${AIPERF_VENV}/bin/aiperf" profile \
    --scenario agentx --url "http://127.0.0.1:${ROUTER_PORT}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --concurrency 48 --benchmark-duration 3600 --tokenizer-trust-remote-code \
    --server-metrics "http://${prefill_ips[0]}:${prefill_ports[0]}/metrics" \
                     "http://${decode_ips[0]}:${decode_ports[0]}/metrics" \
    --output-artifact-dir "${out_dir}" \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    2>&1 | tee "${out_dir}/aiperf.log"
  snapshot_eval_metrics "${out_dir}" after
  test -s "${out_dir}/profile_export_aiperf.json"
  write_aiperf_dashboard_json "${out_dir}/profile_export_aiperf.json" \
    "${RUN_DIR}/benchmark_results/pd-vllm-mooncake-c48.json" 48
}

prepare_mooncake_c48_runtime
