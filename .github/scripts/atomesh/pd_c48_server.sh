# shellcheck shell=bash
# Isolated C48 model-setting experiment; reuse topology/lifecycle only.
start_vllm_server() {
  local role="$1" log_name="$2" port="$3"
  local config="${ATOMESH_SCRIPT_DIR}/../../benchmark/c48-model-gap/${ATOMESH_C48_CASE}.json"
  if [[ "${ATOMESH_C48_CASE}" == "prefix-read-fixed" ]]; then
    config="${ATOMESH_SCRIPT_DIR}/../../benchmark/prefix-read-fixed/config.json"
    python3 "${ATOMESH_SCRIPT_DIR}/pd_prefix_read.py" require-ready || return $?
    export PREFIX_READ_CONTROL_TOKEN="${ATOMESH_RUN_TOKEN:?missing shared run token}"
  fi
  local -a cache_env=()
  build_server_cache_env "${role}" "${port}" cache_env
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" \
    env "${cache_env[@]}" python3 "${ATOMESH_SCRIPT_DIR}/pd_c48_launch.py" \
    --config "${config}" \
    --role "${role}" --model "${MODEL_PATH}" --host "${host_ip}" \
    --proxy "${NODE0_ADDR}" --port "${port}" --discovery "${VLLM_DISCOVERY_PORT}" \
    --output "${RUNTIME_LOG_DIR}/${log_name}.launch.json" --execute
}

run_aiperf_agentic_benchmark() {
  if [[ "${ATOMESH_C48_CASE}" == "prefix-read-fixed" ]]; then
    python3 "${ATOMESH_SCRIPT_DIR}/pd_prefix_read.py" require-ready || return $?
    export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=10
  fi
  ensure_aiperf
  "${AIPERF_VENV}/bin/python" -m pip install 'datasets>=4.7.0'
  local out_dir="${RUN_DIR}/benchmark_results/c48-${ATOMESH_C48_CASE}"
  mkdir -p "${out_dir}"
  git -C "${AIPERF_DIR}" rev-parse HEAD > "${out_dir}/aiperf-commit.txt"
  "${AIPERF_VENV}/bin/python" -m pip freeze > "${out_dir}/client-packages.txt"
  snapshot_eval_metrics "${out_dir}" before
  local -a client=("${AIPERF_VENV}/bin/aiperf")
  if [[ "${ATOMESH_C48_CASE}" == "prefix-read-fixed" ]]; then
    export PREFIX_READ_CLIENT_GATE=1
    export PREFIX_READ_CONTROL_TOKEN="${ATOMESH_RUN_TOKEN:?missing shared run token}"
    export PREFIX_READ_ENDPOINTS="{\"prefill\":\"http://${prefill_ips[0]}:${prefill_ports[0]}\",\"decode\":\"http://${decode_ips[0]}:${decode_ports[0]}\"}"
    export PREFIX_READ_GATE_OUTPUT="${out_dir}/prefix-read-gate.json"
    export PYTHONPATH="${ATOMESH_SCRIPT_DIR}/../../benchmark/prefix-read-fixed/client:${ATOMESH_SCRIPT_DIR}:${PYTHONPATH:-}"
    client=("${AIPERF_VENV}/bin/python" "${ATOMESH_SCRIPT_DIR}/pd_prefix_client.py")
  fi
  # Match the historical InferenceX agentx invocation and pinned scenario defaults.
  # Paths and addresses follow this allocation; no template or context override.
  "${client[@]}" profile \
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
    "${RUN_DIR}/benchmark_results/pd-vllm-c48-${ATOMESH_C48_CASE}.json" 48
}
