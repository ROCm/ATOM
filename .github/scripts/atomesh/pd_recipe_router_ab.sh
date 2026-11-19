# shellcheck shell=bash

PR5090_CONTRACT="${ATOMESH_PR5090_CONTRACT:?recipe contract is required}"
PR5090_HELPER="${ATOMESH_SCRIPT_DIR}/pd_recipe_router_ab.py"
ROUTER_READY_PATH=/readiness
ROUTER_ALIVE_PATH=/liveness
ROUTER_POLICY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["router_policy"][sys.argv[2]])' "${PR5090_CONTRACT}" "${ATOMESH_EXECUTION_PHASE}")"
[[ "${ATOMESH_EXECUTION_PHASE}" == baseline ]] && ROUTER_READY_PATH=/health

start_vllm_server() {
  local role="$1" log_name="$2" server_port="$3"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" \
    python3 "${PR5090_HELPER}" worker --contract "${PR5090_CONTRACT}" \
    --role "${role}" --model-path "${MODEL_PATH}" --port "${server_port}" \
    --host "${host_ip}" --gpus "${HIP_VISIBLE_DEVICES}" \
    --output "${RUNTIME_LOG_DIR}/${log_name}.launch.json"
}

start_router() {
  router_pid=""
  local request="${RUN_DIR}/mooncake-router-${ATOMESH_EXECUTION_PHASE}.args" i
  {
    for i in "${!prefill_ips[@]}"; do
      printf '%s\n' --prefill "http://${prefill_ips[i]}:${prefill_ports[i]}" 7200
    done
    for i in "${!decode_ips[@]}"; do
      printf '%s\n' --decode "http://${decode_ips[i]}:${decode_ports[i]}"
    done
  } > "${request}.tmp"
  mv "${request}.tmp" "${request}"
}

run_aiperf_agentic_benchmark() {
  local output="${RUN_DIR}/benchmark_results/${ATOMESH_EXECUTION_PHASE}" i result
  local -a workers=()
  for i in "${!prefill_ips[@]}"; do workers+=("http://${prefill_ips[i]}:${prefill_ports[i]}"); done
  for i in "${!decode_ips[@]}"; do workers+=("http://${decode_ips[i]}:${decode_ports[i]}"); done
  python3 "${PR5090_HELPER}" benchmark --contract "${PR5090_CONTRACT}" \
    --model-path "${MODEL_PATH}" --router "http://127.0.0.1:${ROUTER_PORT}" \
    --workers "${workers[@]}" --output "${output}" || return $?
  result="$(find "${output}/aiperf_artifacts" -name profile_export_aiperf.json -print -quit)"
  [[ -n "${result}" ]] || { echo 'Missing AgentX result' >&2; return 1; }
  write_aiperf_dashboard_json "${result}" \
    "${RUN_DIR}/benchmark_results/router-ab-${ATOMESH_EXECUTION_PHASE}.json" "${CONC_LIST}"
}
