#!/usr/bin/env bash
# Sourced by pd_server_atom.sh. All artifacts are separate from AIPerf exports.
OBS_ENABLED="${ATOMESH_OBSERVABILITY:-0}"
OBS_DIR="${RUN_DIR}/observability/${ATOMESH_EXECUTION_PHASE}"
OBS_SCRIPT="${ATOMESH_SCRIPT_DIR}/observability.py"
OBS_GPU_PORT=$((19221 + ATOMESH_SERVICE_PORT_OFFSET))
obs_vm_pid=""
obs_agent_pid=""
obs_gpu_pid=""
obs_started=0

obs_stop_process() {
  local pid="$1" deadline=$((SECONDS + 20))
  kill -TERM "${pid}" 2>/dev/null || true
  while process_is_running "${pid}"; do
    if (( SECONDS >= deadline )); then
      echo "[obs] pid=${pid} did not stop within 20s" >&2
      kill -KILL "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 1
    fi
    sleep 0.1
  done
  wait "${pid}"
}

obs_prepare() {
  [[ "${OBS_ENABLED}" == "1" ]] || return 0
  mkdir -p "${OBS_DIR}/events"
  export ATOM_METRICS_REFRESH_INTERVAL_SECONDS=1
  export ATOM_SCHEDULING_METRICS_INTERVAL_SECONDS=0.1
  export ATOM_OBSERVABILITY_RUN_ID="${SLURM_JOB_ID:-local}"
  # One exporter per node, independent of the P/D processes' GPU masks.
  python3 "${OBS_SCRIPT}" gpu --port "${OBS_GPU_PORT}" \
    > "${OBS_DIR}/gpu-rank-${NODE_RANK}.log" 2>&1 &
  obs_gpu_pid=$!
}

obs_server_env() {
  local role="$1" port="$2"
  if [[ "${OBS_ENABLED}" == "1" && "${ATOMESH_OBSERVABILITY_EVENTS:-1}" == "1" ]]; then
    export ATOM_REQUEST_EVENTS_PATH="${OBS_DIR}/events/${role}-rank-${NODE_RANK}-port-${port}.jsonl"
  else
    unset ATOM_REQUEST_EVENTS_PATH
  fi
}

obs_install_binary() {
  local name="$1" archive="$2"
  if command -v "${name}" >/dev/null; then
    command -v "${name}"
    return
  fi
  local bin_dir="${TMPDIR:-/tmp}/atomesh-observability-${SLURM_JOB_ID:-local}/bin"
  mkdir -p "${bin_dir}"
  if [[ ! -x "${bin_dir}/${name}" ]]; then
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 180 \
      "https://github.com/VictoriaMetrics/VictoriaMetrics/releases/download/v1.111.0/${archive}" \
      -o "${bin_dir}/${archive}" >&2
    tar -xzf "${bin_dir}/${archive}" -C "${bin_dir}" "${name}"
  fi
  printf '%s\n' "${bin_dir}/${name}"
}

obs_start() {
  [[ "${OBS_ENABLED}" == "1" && "${NODE_RANK}" -eq 0 ]] || return 0
  local vm agent idx
  vm="$(obs_install_binary victoria-metrics-prod victoria-metrics-linux-amd64-v1.111.0.tar.gz)"
  agent="$(obs_install_binary vmagent-prod vmutils-linux-amd64-v1.111.0.tar.gz)"
  local -a targets=()
  for idx in "${!prefill_ips[@]}"; do
    targets+=(--target "prefill=http://${prefill_ips[$idx]}:${prefill_ports[$idx]}/metrics")
  done
  for idx in "${!decode_ips[@]}"; do
    targets+=(--target "decode=http://${decode_ips[$idx]}:${decode_ports[$idx]}/metrics")
  done
  targets+=(--target "router=http://${NODE0_ADDR}:${PROMETHEUS_PORT}/metrics")
  local ip
  IFS=',' read -r -a obs_ips <<< "${IPADDRS}"
  for ip in "${obs_ips[@]}"; do
    targets+=(--target "gpu=http://${ip}:${OBS_GPU_PORT}/metrics")
  done
  python3 "${OBS_SCRIPT}" prepare --directory "${OBS_DIR}" \
    --run-id "${SLURM_JOB_ID:-local}" --model "${MODEL_NAME}" \
    --topology "${DISPLAY_TOPOLOGY}" --case "${ATOMESH_CELL_ID:-local}" \
    --phase "${ATOMESH_EXECUTION_PHASE}" --events "${ATOMESH_OBSERVABILITY_EVENTS:-1}" "${targets[@]}"
  OBS_VM_PORT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["vm_port"])' "${OBS_DIR}/run.json")"
  "${vm}" -httpListenAddr="127.0.0.1:${OBS_VM_PORT}" \
    -storageDataPath="${OBS_DIR}/vmdata" -retentionPeriod=1d \
    -memory.allowedBytes=256MiB -search.latencyOffset=0 \
    -search.maxPointsPerTimeseries=1000000 \
    > "${OBS_DIR}/victoriametrics.log" 2>&1 &
  obs_vm_pid=$!
  wait_http "http://127.0.0.1:${OBS_VM_PORT}/health" "observability-vm" 60 "${obs_vm_pid}"
  local -a remote_args=(-remoteWrite.url="http://127.0.0.1:${OBS_VM_PORT}/api/v1/write")
  if [[ -n "${ATOMESH_VM_REMOTE_WRITE_URL:-}" ]]; then
    remote_args+=(-remoteWrite.url="${ATOMESH_VM_REMOTE_WRITE_URL}")
  fi
  "${agent}" -httpListenAddr=127.0.0.1:0 \
    -memory.allowedBytes=128MiB \
    -promscrape.config="${OBS_DIR}/scrape.json" \
    "${remote_args[@]}" -remoteWrite.flushInterval=1s \
    -remoteWrite.tmpDataPath="${OBS_DIR}/vmagent-buffer" \
    -remoteWrite.maxDiskUsagePerURL=1GB \
    > "${OBS_DIR}/vmagent.log" 2>&1 &
  obs_agent_pid=$!
  obs_started=1
  # Validate every endpoint before spending an hour on the agentic workload.
  python3 "${OBS_SCRIPT}" preflight --directory "${OBS_DIR}"
  echo "[obs] collecting run=${SLURM_JOB_ID:-local} into ${OBS_DIR}"
}

obs_finish() {
  [[ "${obs_started}" == "1" ]] || return 0
  obs_started=0
  local rc=0
  # Capture the last completed requests before stopping the 1-second scraper.
  sleep 2
  # vmagent flushes pending writes on SIGTERM. Export while VM is still alive.
  obs_stop_process "${obs_agent_pid}" || rc=1
  obs_agent_pid=""
  python3 "${OBS_SCRIPT}" finish --directory "${OBS_DIR}" || rc=1
  obs_stop_process "${obs_vm_pid}" || rc=1
  obs_vm_pid=""
  return "${rc}"
}

obs_cleanup() {
  local pid
  # On benchmark failure, still retain/export partial data and a failure report.
  obs_finish || true
  for pid in "${obs_agent_pid}" "${obs_vm_pid}" "${obs_gpu_pid}"; do
    if [[ -n "${pid}" ]]; then
      obs_stop_process "${pid}" || true
    fi
  done
  obs_agent_pid="" obs_vm_pid="" obs_gpu_pid=""
}
