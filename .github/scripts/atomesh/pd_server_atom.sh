#!/usr/bin/env bash
set -euo pipefail

ATOMESH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODE_RANK="${NODE_RANK:-0}"
NODE0_ADDR="${NODE0_ADDR:-127.0.0.1}"
IPADDRS="${IPADDRS:-127.0.0.1}"
RUN_DIR="${RUN_DIR:-/run_logs/slurm_job-${SLURM_JOB_ID:-local}}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
BACKEND="${BACKEND:-atom}"
TOPOLOGY="${TOPOLOGY:-unknown}"
DISPLAY_TOPOLOGY="${DISPLAY_TOPOLOGY:-${TOPOLOGY}}"
ATOMESH_PD_WORKER_LAYOUT="${ATOMESH_PD_WORKER_LAYOUT:-multi_node}"
SINGLE_NODE_PD=0
PREFILL_SINGLE_NODE_PD=0
DECODE_SINGLE_NODE_PD=0
case "${ATOMESH_PD_WORKER_LAYOUT}" in
  single_node)
    SINGLE_NODE_PD=1
    ;;
  prefill_single_node)
    PREFILL_SINGLE_NODE_PD=1
    ;;
  decode_single_node)
    DECODE_SINGLE_NODE_PD=1
    ;;
esac

xP="${xP:-1}"
yD="${yD:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"

PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
ROUTER_POLICY="${ROUTER_POLICY:-random}"
ATOM_PD_RANK_MAPPING_POLICY="${ATOM_PD_RANK_MAPPING_POLICY:-none}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29100}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"
PREFILL_DP_MASTER_PORT="${PREFILL_DP_MASTER_PORT:-29500}"
PREFILL_DP_BASE_PORT="${PREFILL_DP_BASE_PORT:-29600}"
DECODE_DP_MASTER_PORT="${DECODE_DP_MASTER_PORT:-29700}"
DECODE_DP_BASE_PORT="${DECODE_DP_BASE_PORT:-29800}"
ATOMESH_EXECUTION_PHASE="${ATOMESH_EXECUTION_PHASE:-combined}"
ATOMESH_SERVICE_PORT_OFFSET="${ATOMESH_SERVICE_PORT_OFFSET:-0}"
case "${ATOMESH_EXECUTION_PHASE}" in
  combined|benchmark|eval) ;;
  *)
    echo "ERROR: unsupported ATOMESH_EXECUTION_PHASE=${ATOMESH_EXECUTION_PHASE}" >&2
    exit 2
    ;;
esac
if [[ ! "${ATOMESH_SERVICE_PORT_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ATOMESH_SERVICE_PORT_OFFSET must be a non-negative integer" >&2
  exit 2
fi
PREFILL_PORT=$((PREFILL_PORT + ATOMESH_SERVICE_PORT_OFFSET))
DECODE_PORT=$((DECODE_PORT + ATOMESH_SERVICE_PORT_OFFSET))
ROUTER_PORT=$((ROUTER_PORT + ATOMESH_SERVICE_PORT_OFFSET))
PROMETHEUS_PORT=$((PROMETHEUS_PORT + ATOMESH_SERVICE_PORT_OFFSET))
HANDSHAKE_PORT=$((HANDSHAKE_PORT + ATOMESH_SERVICE_PORT_OFFSET))
PREFILL_DP_MASTER_PORT=$((PREFILL_DP_MASTER_PORT + ATOMESH_SERVICE_PORT_OFFSET))
PREFILL_DP_BASE_PORT=$((PREFILL_DP_BASE_PORT + ATOMESH_SERVICE_PORT_OFFSET))
DECODE_DP_MASTER_PORT=$((DECODE_DP_MASTER_PORT + ATOMESH_SERVICE_PORT_OFFSET))
DECODE_DP_BASE_PORT=$((DECODE_DP_BASE_PORT + ATOMESH_SERVICE_PORT_OFFSET))
validate_shifted_port() {
  local name="$1"
  local value="${!name}"
  if (( value < 1 || value > 65535 )); then
    echo "ERROR: ${name}=${value} is outside the valid TCP/UDP port range" >&2
    exit 2
  fi
}
for shifted_port_name in \
  PREFILL_PORT \
  DECODE_PORT \
  ROUTER_PORT \
  PROMETHEUS_PORT \
  HANDSHAKE_PORT \
  PREFILL_DP_MASTER_PORT \
  PREFILL_DP_BASE_PORT \
  DECODE_DP_MASTER_PORT \
  DECODE_DP_BASE_PORT; do
  validate_shifted_port "${shifted_port_name}"
done
unset shifted_port_name
unset -f validate_shifted_port
USE_EXPLICIT_DP_PORTS=0
if [[ "${SINGLE_NODE_PD}" == "1" || "${PREFILL_SINGLE_NODE_PD}" == "1" || "${DECODE_SINGLE_NODE_PD}" == "1" ]]; then
  USE_EXPLICIT_DP_PORTS=1
fi

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS:-}"
ONLINE_QUANT_CONFIG="${ONLINE_QUANT_CONFIG:-}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
SPEC_METHOD="${SPEC_METHOD:-}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
PREFILL_EXTRA_SERVER_ARGS="${PREFILL_EXTRA_SERVER_ARGS:-}"
DECODE_EXTRA_SERVER_ARGS="${DECODE_EXTRA_SERVER_ARGS:-}"
PREFILL_SERVER_ARGS="${EXTRA_SERVER_ARGS}"
DECODE_SERVER_ARGS="${EXTRA_SERVER_ARGS}"
if [[ -n "${PREFILL_EXTRA_SERVER_ARGS}" ]]; then
  PREFILL_SERVER_ARGS="${PREFILL_SERVER_ARGS:+${PREFILL_SERVER_ARGS} }${PREFILL_EXTRA_SERVER_ARGS}"
fi
if [[ -n "${DECODE_EXTRA_SERVER_ARGS}" ]]; then
  DECODE_SERVER_ARGS="${DECODE_SERVER_ARGS:+${DECODE_SERVER_ARGS} }${DECODE_EXTRA_SERVER_ARGS}"
fi

has_cli_flag() {
  local args="$1"
  local flag="$2"
  [[ " ${args} " == *" ${flag} "* ]]
}

is_agentic_dpa() {
  [[ "${BENCHMARK_KIND}" == "aiperf_agentic" ]] \
    && {
      has_cli_flag "${PREFILL_EXTRA_SERVER_ARGS}" "--enable-dp-attention" \
        || has_cli_flag "${DECODE_EXTRA_SERVER_ARGS}" "--enable-dp-attention"
    }
}

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-4,8}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-${CONC_LIST//,/x}}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

RUN_EVAL="${RUN_EVAL:-false}"
EVAL_TASK="${EVAL_TASK:-gsm8k}"
EVAL_FEWSHOT="${EVAL_FEWSHOT:-3}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_MODEL_TYPE="${EVAL_MODEL_TYPE:-local-completions}"
EVAL_ENDPOINT="${EVAL_ENDPOINT:-completions}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EVAL_MAX_GEN_TOKS="${EVAL_MAX_GEN_TOKS:-}"
EVAL_APPLY_CHAT_TEMPLATE="${EVAL_APPLY_CHAT_TEMPLATE:-false}"
EVAL_FEWSHOT_AS_MULTITURN="${EVAL_FEWSHOT_AS_MULTITURN:-false}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-}"

# SWE-bench Lite runs entirely from ATOM-owned scripts. Agent generation and
# official scoring both use the Docker daemon mounted into the rank-0 container.
SWEBENCH_VENV="${SWEBENCH_VENV:-/tmp/atomesh-swebench-venv-${SLURM_JOB_ID:-local}}"
SWEBENCH_AGENT_WORKERS="${SWEBENCH_AGENT_WORKERS:-32}"
SWEBENCH_AGENT_STEP_LIMIT="${SWEBENCH_AGENT_STEP_LIMIT:-150}"
SWEBENCH_CASE_TIMEOUT="${SWEBENCH_CASE_TIMEOUT:-3600}"
SWEBENCH_AGENT_TIMEOUT="${SWEBENCH_AGENT_TIMEOUT:-21600}"
SWEBENCH_SCORE_TIMEOUT="${SWEBENCH_SCORE_TIMEOUT:-7200}"
SWEBENCH_MAX_WORKERS="${SWEBENCH_MAX_WORKERS:-4}"
SWEBENCH_EVAL_TIMEOUT="${SWEBENCH_EVAL_TIMEOUT:-900}"
# The host daemon is shared with every other job on the node: refuse to start
# without image headroom, and hand the pulled images back when the run ends.
SWEBENCH_MIN_DISK_GB="${SWEBENCH_MIN_DISK_GB:-150}"
SWEBENCH_PRUNE_IMAGES="${SWEBENCH_PRUNE_IMAGES:-true}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-5000}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

BENCHMARK_KIND="${BENCHMARK_KIND:-random}"
AIPERF_DIR="${AIPERF_DIR:-/tmp/atomesh-aiperf}"
AIPERF_VENV="${AIPERF_VENV:-/tmp/atomesh-aiperf-venv}"
AIPERF_COMMIT="${AIPERF_COMMIT:-b7b16cf851885567988a643282266bce74e34437}"
AIPERF_SCENARIO="${AIPERF_SCENARIO:-inferencex-agentx-mvp}"
AIPERF_PUBLIC_DATASET="${AIPERF_PUBLIC_DATASET:-semianalysis_cc_traces_weka_062126_256k}"
AIPERF_MAX_CONTEXT_LENGTH="${AIPERF_MAX_CONTEXT_LENGTH:-262144}"
AIPERF_NUM_DATASET_ENTRIES="${AIPERF_NUM_DATASET_ENTRIES:-393}"
AIPERF_BENCHMARK_DURATION="${AIPERF_BENCHMARK_DURATION:-1800}"
AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
AIPERF_WARMUP_GRACE_PERIOD="${AIPERF_WARMUP_GRACE_PERIOD:-1800}"
AIPERF_TRAJECTORY_START_MIN_RATIO="${AIPERF_TRAJECTORY_START_MIN_RATIO:-0.25}"
AIPERF_TRAJECTORY_START_MAX_RATIO="${AIPERF_TRAJECTORY_START_MAX_RATIO:-0.75}"
AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.50}"
AIPERF_SLICE_DURATION="${AIPERF_SLICE_DURATION:-1.0}"
AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT="${AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT:-300}"
AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT:-900000}"
AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"
AIPERF_UNSAFE_OVERRIDE="${AIPERF_UNSAFE_OVERRIDE:-}"
PREFILL_KV_TRANSFER_CONFIG="${PREFILL_KV_TRANSFER_CONFIG:-}"
DECODE_KV_TRANSFER_CONFIG="${DECODE_KV_TRANSFER_CONFIG:-}"

default_profiler_dir="${RUN_DIR}/online_quant/rank-${NODE_RANK}"
if [[ "${ATOMESH_EXECUTION_PHASE}" != "combined" ]]; then
  default_profiler_dir="${RUN_DIR}/online_quant/${ATOMESH_EXECUTION_PHASE}/rank-${NODE_RANK}"
fi
export ATOM_TORCH_PROFILER_DIR="${ATOM_TORCH_PROFILER_DIR:-${default_profiler_dir}}"
RUNTIME_LOG_DIR="${RUN_DIR}/logs"
if [[ "${ATOMESH_EXECUTION_PHASE}" != "combined" ]]; then
  RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR}/${ATOMESH_EXECUTION_PHASE}"
fi
mkdir -p "${RUNTIME_LOG_DIR}" "${RUN_DIR}"/{benchmark_results,eval_results} "${ATOM_TORCH_PROFILER_DIR}"

role_tp="${PREFILL_TP_SIZE}"
if [[ "${PREFILL_SINGLE_NODE_PD}" == "1" && "${NODE_RANK}" -gt 0 ]]; then
  role_tp="${DECODE_TP_SIZE}"
elif [[ "${NODE_RANK}" -ge "${xP}" ]]; then
  role_tp="${DECODE_TP_SIZE}"
fi
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$(seq -s, 0 "$((role_tp - 1))")"
fi
rm -rf /root/.cache/atom/* 2>/dev/null || true
echo "[runtime] phase=${ATOMESH_EXECUTION_PHASE} service_port_offset=${ATOMESH_SERVICE_PORT_OFFSET}"
echo "[runtime] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"

dump_launch_info() {
  local role="$1"
  shift
  echo ""
  echo "========================================"
  echo "  ${role} launch info"
  echo "========================================"
  echo "--- environment ---"
  env | grep -E '^(HIP_|HSA_|AITER_|ATOM_|RCCL_|NCCL_|CUDA_|MOONCAKE_|UCX_)' | sort || true
  echo "--- command ---"
  printf '%q ' "$@"
  echo ""
  echo "========================================"
  echo ""
}

apply_prefixed_env() {
  local prefix="$1"
  local role_ip="$2"
  local name raw value
  while IFS='=' read -r name raw; do
    [[ "${name}" == "${prefix}"* ]] || continue
    value="${raw//\$\{ROLE_IP\}/${role_ip}}"
    value="${value//\$\{HANDSHAKE_PORT\}/${HANDSHAKE_PORT}}"
    export "${name#${prefix}}=${value}"
  done < <(env)
}

# Names the last apply_role_env() exported for a role.
ROLE_ENV_NAMES=()

# Both servers are launched from this one shell, so a variable exported for one
# role stays in the environment the next role inherits. Only names the two roles
# both define get overwritten; a prefill-only name reaches decode unchanged --
# VLLM_PP_LAYER_PARTITION from a pp4 prefill aborts a pp1 decode's model build
# with "len(partitions)=4 does not match pp_size=1". Drop the previous role's
# names before applying this one's.
apply_role_env() {
  local prefix="$1"
  local role_ip="$2"
  local name
  for name in ${ROLE_ENV_NAMES[@]+"${ROLE_ENV_NAMES[@]}"}; do
    unset "${name}"
  done
  ROLE_ENV_NAMES=()
  while IFS='=' read -r name _; do
    [[ "${name}" == "${prefix}"* ]] || continue
    ROLE_ENV_NAMES+=("${name#${prefix}}")
  done < <(env)
  # A name the common block also sets was just unset with the previous role's,
  # so put the common value back before the role overrides it.
  apply_prefixed_env "ATOMESH_ENV_" "${role_ip}"
  apply_prefixed_env "${prefix}" "${role_ip}"
}

host_ip="$(echo "${IPADDRS}" | tr ',' '\n' | sed -n "$((NODE_RANK + 1))p")"
if [[ -z "${host_ip}" ]]; then
  host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
host_name="$(hostname)"

apply_prefixed_env "ATOMESH_ENV_" "${host_ip}"

IFS=',' read -r -a IP_ARRAY <<< "${IPADDRS}"

prefill_args=()
prefill_ips=()
prefill_ports=()
decode_args=()
decode_ips=()
decode_ports=()
if [[ "${SINGLE_NODE_PD}" == "1" ]]; then
  if [[ "${xP}" != "1" || "${yD}" != "1" ]]; then
    echo "ERROR: single_node PD worker layout currently supports only 1 prefill and 1 decode worker" >&2
    exit 1
  fi
  prefill_ips+=("${IP_ARRAY[0]}")
  prefill_ports+=("${PREFILL_PORT}")
  prefill_args+=(--prefill "http://${IP_ARRAY[0]}:${PREFILL_PORT}")
  decode_ips+=("${IP_ARRAY[0]}")
  decode_ports+=("${DECODE_PORT}")
  decode_args+=(--decode "http://${IP_ARRAY[0]}:${DECODE_PORT}")
elif [[ "${PREFILL_SINGLE_NODE_PD}" == "1" ]]; then
  for idx in $(seq 0 $((xP - 1))); do
    prefill_port=$((PREFILL_PORT + idx))
    prefill_ips+=("${IP_ARRAY[0]}")
    prefill_ports+=("${prefill_port}")
    prefill_args+=(--prefill "http://${IP_ARRAY[0]}:${prefill_port}")
  done

  for idx in $(seq 0 $((yD - 1))); do
    node_idx=$((1 + idx))
    decode_ips+=("${IP_ARRAY[$node_idx]}")
    decode_ports+=("${DECODE_PORT}")
    decode_args+=(--decode "http://${IP_ARRAY[$node_idx]}:${DECODE_PORT}")
  done
elif [[ "${DECODE_SINGLE_NODE_PD}" == "1" ]]; then
  for idx in $(seq 0 $((xP - 1))); do
    prefill_ips+=("${IP_ARRAY[$idx]}")
    prefill_ports+=("${PREFILL_PORT}")
    prefill_args+=(--prefill "http://${IP_ARRAY[$idx]}:${PREFILL_PORT}")
  done

  decode_node_idx="${xP}"
  for idx in $(seq 0 $((yD - 1))); do
    decode_port=$((DECODE_PORT + idx))
    decode_ips+=("${IP_ARRAY[$decode_node_idx]}")
    decode_ports+=("${decode_port}")
    decode_args+=(--decode "http://${IP_ARRAY[$decode_node_idx]}:${decode_port}")
  done
else
  for idx in $(seq 0 $((xP - 1))); do
    prefill_ips+=("${IP_ARRAY[$idx]}")
    prefill_ports+=("${PREFILL_PORT}")
    prefill_args+=(--prefill "http://${IP_ARRAY[$idx]}:${PREFILL_PORT}")
  done

  for idx in $(seq 0 $((yD - 1))); do
    node_idx=$((xP + idx))
    decode_ips+=("${IP_ARRAY[$node_idx]}")
    decode_ports+=("${DECODE_PORT}")
    decode_args+=(--decode "http://${IP_ARRAY[$node_idx]}:${DECODE_PORT}")
  done
fi

prefill_parallel=(-tp "${PREFILL_TP_SIZE}")
if [[ "${PREFILL_ENABLE_DP}" == "true" ]]; then
  prefill_parallel+=("--enable-dp-attention")
fi

decode_parallel=(-tp "${DECODE_TP_SIZE}")
if [[ "${DECODE_ENABLE_DP}" == "true" ]]; then
  decode_parallel+=("--enable-dp-attention")
fi

build_cudagraph_args() {
  local value="$1"
  local -n out="$2"
  case "${value:-}" in
    ""|none|None|NONE|false|False|FALSE|off|Off|OFF|disabled|Disabled|DISABLED)
      out=()
      ;;
    *)
      out=(--cudagraph-capture-sizes "${value}")
      ;;
  esac
}

prefill_cudagraph_args=()
decode_cudagraph_args=()
build_cudagraph_args "${PREFILL_CUDAGRAPH:-}" prefill_cudagraph_args
build_cudagraph_args "${DECODE_CUDAGRAPH:-}" decode_cudagraph_args

build_server_cache_env() {
  local role="$1"
  local server_port="$2"
  local -n out="$3"
  local cache_base cache_root

  cache_base="${ATOMESH_WORKER_CACHE_BASE:-${XDG_CACHE_HOME:-/tmp/atomesh-cache-${SLURM_JOB_ID:-local}-${NODE_RANK}}/workers}"
  cache_root="${cache_base}/${role}-${server_port}"
  mkdir -p "${cache_root}"/{home,xdg,torchinductor,triton,aiter/jit,flydsl}

  out=(
    "HOME=${cache_root}/home"
    "XDG_CACHE_HOME=${cache_root}/xdg"
    "TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor"
    "TRITON_CACHE_DIR=${cache_root}/triton"
    "AITER_CACHE_DIR=${cache_root}/aiter"
    "AITER_JIT_DIR=${cache_root}/aiter/jit"
    "FLYDSL_RUNTIME_CACHE_DIR=${cache_root}/flydsl"
  )
  echo "[runtime] ${role} cache root=${cache_root} (port=${server_port})"
}

server_common=(
  --model "${MODEL_PATH}"
  --host 0.0.0.0
  --trust-remote-code
  --kv_cache_dtype "${KV_CACHE_DTYPE}"
  --block-size "${BLOCK_SIZE}"
  --gpu-memory-utilization "${MEM_FRACTION}"
)

if [[ "${ENABLE_PREFIX_CACHING}" != "true" && "${ENABLE_PREFIX_CACHING}" != "1" ]]; then
  server_common+=(--no-enable_prefix_caching)
fi

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  server_common+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  server_common+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${ONLINE_QUANT_CONFIG}" ]]; then
  server_common+=(--online_quant_config "${ONLINE_QUANT_CONFIG}")
fi
if [[ -n "${HF_OVERRIDES}" ]]; then
  server_common+=(--hf-overrides "${HF_OVERRIDES}")
fi
if [[ -n "${SPEC_METHOD}" ]]; then
  server_common+=(--method "${SPEC_METHOD}")
fi
if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
  server_common+=(--draft-model "${DRAFT_MODEL_PATH}")
fi
if [[ -n "${NUM_SPEC_TOKENS}" ]]; then
  server_common+=(--num-speculative-tokens "${NUM_SPEC_TOKENS}")
fi

wait_http() {
  local url="$1"
  local name="$2"
  local timeout="$3"
  local pid="${4:-}"
  local deadline=$(( $(date +%s) + timeout ))
  echo "[wait] ${name} ${url} timeout=${timeout}s"
  until curl -sf --max-time 10 "${url}" >/dev/null 2>&1; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      set +e
      wait "${pid}"
      local rc=$?
      set -e
      [[ "${rc}" -eq 0 ]] && rc=1
      echo "[wait][FAIL] ${name} process exited before becoming ready rc=${rc}" >&2
      exit "${rc}"
    fi
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "[wait][FAIL] ${name} not ready after ${timeout}s" >&2
      exit 1
    fi
    sleep 10
  done
  echo "[wait][OK] ${name}"
}

wait_router_closed() {
  local miss_count=0
  local max_misses=3
  echo "[wait] router shutdown http://${NODE0_ADDR}:${ROUTER_PORT}/health"
  while true; do
    if curl -sf --max-time 10 "http://${NODE0_ADDR}:${ROUTER_PORT}/health" >/dev/null 2>&1; then
      miss_count=0
      if [[ -n "${server_pid:-}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
        set +e
        wait "${server_pid}"
        local rc=$?
        set -e
        [[ "${rc}" -eq 0 ]] && rc=1
        echo "[wait][FAIL] worker process exited while router was still alive rc=${rc}" >&2
        exit "${rc}"
      fi
    else
      miss_count=$((miss_count + 1))
      if [[ "${miss_count}" -ge "${max_misses}" ]]; then
        break
      fi
      echo "[wait] router health miss ${miss_count}/${max_misses}; continuing"
    fi
    sleep 10
  done
  echo "[wait][OK] router closed"
}

start_logged_process() {
  local pid_var="$1"
  local log_file="$2"
  shift 2

  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" > >(tee "${log_file}") 2>&1 &
  else
    "$@" > >(tee "${log_file}") 2>&1 &
  fi
  printf -v "${pid_var}" '%s' "$!"
}

process_is_running() {
  local pid="$1"
  local state

  if [[ -r "/proc/${pid}/stat" ]]; then
    state="$(awk '{ print $3 }' "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ -n "${state}" && "${state}" != "Z" ]]
    return
  fi

  kill -0 "${pid}" 2>/dev/null
}

terminate_process_group() {
  local pid="${1:-}"
  local deadline

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  process_is_running "${pid}" || return 0

  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  deadline=$(( $(date +%s) + 20 ))
  while process_is_running "${pid}" && [[ "$(date +%s)" -lt "${deadline}" ]]; do
    sleep 1
  done
  if process_is_running "${pid}"; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup_processes() {
  local rc=$?
  local pid
  for pid in "$@"; do
    terminate_process_group "${pid}"
  done
  return "${rc}"
}

write_metadata() {
  local metadata_file="${RUN_DIR}/metadata-rank-${NODE_RANK}.json"
  if [[ "${ATOMESH_EXECUTION_PHASE}" != "combined" ]]; then
    metadata_file="${RUN_DIR}/metadata-rank-${NODE_RANK}-${ATOMESH_EXECUTION_PHASE}.json"
  fi
  cat > "${metadata_file}" <<EOF
{
  "rank": ${NODE_RANK},
  "execution_phase": "${ATOMESH_EXECUTION_PHASE}",
  "host": "${host_name}",
  "ip": "${host_ip}",
  "model": "${MODEL_NAME}",
  "model_path": "${MODEL_PATH}",
  "backend": "${BACKEND}",
  "topology": "${TOPOLOGY}",
  "display_topology": "${DISPLAY_TOPOLOGY}",
  "pd_worker_layout": "${ATOMESH_PD_WORKER_LAYOUT}",
  "prefill_ips": "$(IFS=,; echo "${prefill_ips[*]}")",
  "prefill_ports": "$(IFS=,; echo "${prefill_ports[*]}")",
  "decode_ips": "$(IFS=,; echo "${decode_ips[*]}")",
  "decode_ports": "$(IFS=,; echo "${decode_ports[*]}")"
}
EOF
}

start_prefill() {
  local log_name="$1"
  local server_port="${2:-${PREFILL_PORT}}"
  local handshake_port="${3:-${HANDSHAKE_PORT}}"
  local dp_master_port="${4:-${PREFILL_DP_MASTER_PORT}}"
  local dp_base_port="${5:-${PREFILL_DP_BASE_PORT}}"
  apply_role_env "ATOMESH_PREFILL_ENV_" "${host_ip}"
  local -a prefill_cache_env=()
  build_server_cache_env "prefill" "${server_port}" prefill_cache_env
  local -a prefill_dp_env=()
  if [[ "${USE_EXPLICIT_DP_PORTS}" == "1" ]]; then
    prefill_dp_env=(
      "ATOM_DP_MASTER_PORT=${dp_master_port}"
      "ATOM_DP_BASE_PORT=${dp_base_port}"
    )
  fi
  local prefill_kv_transfer_config
  if [[ -n "${PREFILL_KV_TRANSFER_CONFIG}" ]]; then
    prefill_kv_transfer_config="${PREFILL_KV_TRANSFER_CONFIG}"
  else
    prefill_kv_transfer_config="{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${handshake_port}}"
  fi
  echo "[prefill] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} handshake=${handshake_port} dp_master=${dp_master_port} dp_base=${dp_base_port} cudagraph=${PREFILL_CUDAGRAPH:-none}"
  local -a prefill_cmd=(
    python3 -m atom.entrypoints.openai_server
    "${server_common[@]}"
    --server-port "${server_port}"
    "${prefill_parallel[@]}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --kv-transfer-config "${prefill_kv_transfer_config}"
    "${prefill_cudagraph_args[@]}"
    ${PREFILL_SERVER_ARGS}
  )
  dump_launch_info "PREFILL" "${prefill_cmd[@]}"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" env "${prefill_cache_env[@]}" "${prefill_dp_env[@]}" "${prefill_cmd[@]}"
}

start_decode() {
  local log_name="${1:-decode-rank-${NODE_RANK}}"
  local server_port="${2:-${DECODE_PORT}}"
  local handshake_port="${3:-${HANDSHAKE_PORT}}"
  local dp_master_port="${4:-${DECODE_DP_MASTER_PORT}}"
  local dp_base_port="${5:-${DECODE_DP_BASE_PORT}}"
  apply_role_env "ATOMESH_DECODE_ENV_" "${host_ip}"
  local max_conc
  max_conc="$(echo "${BENCH_MAX_CONCURRENCY}" | tr 'x,' '\n' | sort -n | tail -1)"
  local decode_max_num_seqs="${MAX_NUM_SEQS}"
  if [[ -n "${DECODE_MAX_NUM_SEQS}" ]]; then
    decode_max_num_seqs="${DECODE_MAX_NUM_SEQS}"
  fi
  local -a decode_max_num_batched_tokens_args=()
  if [[ -n "${DECODE_MAX_NUM_BATCHED_TOKENS}" ]]; then
    decode_max_num_batched_tokens_args=(
      --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}"
    )
  fi
  if [[ "${ISL_LIST}" == "1024" && "${OSL}" == "1024" ]]; then
    decode_max_num_seqs="${max_conc}"
  fi
  local -a decode_cache_env=()
  build_server_cache_env "decode" "${server_port}" decode_cache_env
  local -a decode_dp_env=()
  if [[ "${USE_EXPLICIT_DP_PORTS}" == "1" ]]; then
    decode_dp_env=(
      "ATOM_DP_MASTER_PORT=${dp_master_port}"
      "ATOM_DP_BASE_PORT=${dp_base_port}"
    )
  fi
  local decode_kv_transfer_config
  if [[ -n "${DECODE_KV_TRANSFER_CONFIG}" ]]; then
    decode_kv_transfer_config="${DECODE_KV_TRANSFER_CONFIG}"
  else
    decode_kv_transfer_config="{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${handshake_port}}"
  fi
  echo "[decode] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} handshake=${handshake_port} dp_master=${dp_master_port} dp_base=${dp_base_port} cudagraph=${DECODE_CUDAGRAPH:-none}"
  local -a decode_cmd=(
    python3 -m atom.entrypoints.openai_server
    "${server_common[@]}"
    --server-port "${server_port}"
    "${decode_parallel[@]}"
    --max-num-seqs "${decode_max_num_seqs}"
    "${decode_max_num_batched_tokens_args[@]}"
    --kv-transfer-config "${decode_kv_transfer_config}"
    "${decode_cudagraph_args[@]}"
    ${DECODE_SERVER_ARGS}
  )
  dump_launch_info "DECODE" "${decode_cmd[@]}"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" env "${decode_cache_env[@]}" "${decode_dp_env[@]}" "${decode_cmd[@]}"
}

start_router() {
  echo "[router] prefill=${prefill_args[*]} decode=${decode_args[*]}"
  case "${ATOM_PD_RANK_MAPPING_POLICY}" in
    none|idx2idx) ;;
    *)
      echo "[router][FAIL] invalid ATOM_PD_RANK_MAPPING_POLICY=${ATOM_PD_RANK_MAPPING_POLICY}" >&2
      exit 1
      ;;
  esac

  local router_policy="${ROUTER_POLICY}"
  local -a router_rank_mapping_args=()
  if [[ "${ATOM_PD_RANK_MAPPING_POLICY}" != "none" ]] \
    && has_cli_flag "${PREFILL_EXTRA_SERVER_ARGS}" "--enable-dp-attention" \
    && has_cli_flag "${DECODE_EXTRA_SERVER_ARGS}" "--enable-dp-attention"; then
    router_rank_mapping_args=(
      --atom-pd-rank-mapping-policy "${ATOM_PD_RANK_MAPPING_POLICY}"
    )
  fi
  local -a router_dp_aware_args=()
  if is_agentic_dpa; then
    router_policy="dp_sticky"
    router_dp_aware_args=(--dp-aware)
  elif [[ "${#router_rank_mapping_args[@]}" -gt 0 ]]; then
    router_dp_aware_args=(--dp-aware)
  fi
  local -a router_cmd=(
    /app/ATOM/atom/mesh/target/release/atomesh launch
    --host 0.0.0.0
    --port "${ROUTER_PORT}"
    --pd-disaggregation
    "${prefill_args[@]}"
    "${decode_args[@]}"
    --policy "${router_policy}"
    "${router_rank_mapping_args[@]}"
    "${router_dp_aware_args[@]}"
    --backend atom
    --log-level info
    --disable-circuit-breaker
    --prometheus-port "${PROMETHEUS_PORT}"
  )
  dump_launch_info "ROUTER" "${router_cmd[@]}"
  start_logged_process router_pid "${RUNTIME_LOG_DIR}/router.log" "${router_cmd[@]}"
}

run_benchmark() {
  if [[ "${BENCHMARK_KIND}" == "aiperf_agentic" ]]; then
    run_aiperf_agentic_benchmark
    return
  fi

  local bench_root="/tmp/atomesh-inferencex"
  local bench_repo_url="https://github.com/SemiAnalysisAI/InferenceX.git"
  local bench_repo_dir="${bench_root}/InferenceX"
  local bench_serving_dir="${bench_repo_dir}/utils/bench_serving"
  local bench_script="${bench_serving_dir}/benchmark_serving.py"
  if [[ ! -f "${bench_script}" ]] || [[ "$(git -C "${bench_repo_dir}" config --get remote.origin.url 2>/dev/null || true)" != "${bench_repo_url}" ]]; then
    rm -rf "${bench_root}"
    mkdir -p "${bench_root}"
    git clone --depth 1 --filter=blob:none --sparse "${bench_repo_url}" "${bench_repo_dir}"
    git -C "${bench_repo_dir}" sparse-checkout set utils/bench_serving
  fi
  IFS=',' read -r -a isls <<< "${ISL_LIST}"
  IFS=',' read -r -a concs <<< "${CONC_LIST}"
  local safe_model="${MODEL_NAME//\//-}"
  for isl in "${isls[@]}"; do
    for conc in "${concs[@]}"; do
      local result_file="pd-${BACKEND}-${safe_model}-${TOPOLOGY}-isl${isl}-osl${OSL}-conc${conc}-${RANDOM_RANGE_RATIO}.json"
      echo "[bench] ${result_file}"
      PYTHONDONTWRITEBYTECODE=1 python "${bench_script}" \
        --model="${MODEL_PATH}" \
        --backend=vllm \
        --base-url="http://127.0.0.1:${ROUTER_PORT}" \
        --dataset-name=random \
        --random-input-len="${isl}" \
        --random-output-len="${OSL}" \
        --random-range-ratio "${RANDOM_RANGE_RATIO}" \
        --num-prompts="$(( conc * BENCH_NUM_PROMPTS_MULTIPLIER ))" \
        --max-concurrency="${conc}" \
        --trust-remote-code \
        --num-warmups="$(( 2 * conc ))" \
        --request-rate="${REQUEST_RATE}" \
        --ignore-eos \
        --save-result \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --result-dir="${RUN_DIR}/benchmark_results" \
        --result-filename="${result_file}"
    done
  done
}

ensure_aiperf() {
  local current_commit=""
  if [[ -d "${AIPERF_DIR}/.git" ]]; then
    current_commit="$(git -C "${AIPERF_DIR}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -x "${AIPERF_VENV}/bin/aiperf" && "${current_commit}" == "${AIPERF_COMMIT}" ]]; then
    return
  fi

  echo "[aiperf] preparing ${AIPERF_DIR} @ ${AIPERF_COMMIT}"
  mkdir -p "$(dirname "${AIPERF_DIR}")" "$(dirname "${AIPERF_VENV}")"
  if [[ ! -d "${AIPERF_DIR}/.git" ]]; then
    rm -rf "${AIPERF_DIR}"
    git clone https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_DIR}"
  fi
  git -C "${AIPERF_DIR}" fetch https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_COMMIT}"
  git -C "${AIPERF_DIR}" checkout --detach "${AIPERF_COMMIT}"
  rm -rf "${AIPERF_VENV}"
  python3 -m venv "${AIPERF_VENV}"
  "${AIPERF_VENV}/bin/python" -m pip install --upgrade pip
  "${AIPERF_VENV}/bin/python" -m pip install -e "${AIPERF_DIR}"
  "${AIPERF_VENV}/bin/aiperf" --version
}

write_aiperf_dashboard_json() {
  local aiperf_json="$1"
  local out_json="$2"
  local conc="$3"
  python3 - "${aiperf_json}" "${out_json}" "${conc}" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
conc = int(sys.argv[3])
data = json.loads(src.read_text(encoding="utf-8"))


def avg(name):
    value = data.get(name)
    if isinstance(value, dict):
        return value.get("avg")
    return value


def pct(name, key):
    value = data.get(name)
    if isinstance(value, dict):
        return value.get(key)
    return None


def total_tokens(name):
    """Return one of AIPerf's profiling-only aggregate token counters."""
    value = avg(name)
    return int(value) if isinstance(value, (int, float)) else None


# These aggregates contain successful profiling records only: AIPerf excludes
# its internal warmup and requests cancelled during grace-period draining.
cache_hit_tokens = total_tokens("total_usage_prompt_cache_read_tokens")
cache_total_tokens = total_tokens("total_usage_prompt_tokens")

payload = {
    "benchmark_backend": "atom",
    # Directory holding this run's profile_export.jsonl, so process_result.py can
    # find the per-request records both interactivity definitions are computed
    # from, without reconstructing the directory name.
    "aiperf_artifact_dir": src.parent.name,
    "benchmark_model_name": os.environ.get("MODEL_NAME")
    or data.get("model")
    or data.get("model_id"),
    "backend": "atom",
    "benchmark_kind": os.environ.get("BENCHMARK_KIND") or "aiperf_agentic",
    "scenario": os.environ.get("AIPERF_SCENARIO"),
    "public_dataset": os.environ.get("AIPERF_PUBLIC_DATASET"),
    "topology": os.environ.get("TOPOLOGY") or data.get("topology"),
    "display_topology": os.environ.get("DISPLAY_TOPOLOGY")
    or data.get("display_topology"),
    "precision": os.environ.get("PRECISION") or data.get("precision"),
    "random_input_len": int(
        data.get("max_context_length")
        or os.environ.get("AIPERF_MAX_CONTEXT_LENGTH")
        or 0
    ),
    "random_output_len": 1024,
    "max_concurrency": conc,
    "random_range_ratio": "",
    "request_throughput": avg("request_throughput"),
    "mean_ttft_ms": avg("time_to_first_token"),
    "median_ttft_ms": pct("time_to_first_token", "p50"),
    "p99_ttft_ms": pct("time_to_first_token", "p99"),
    "mean_itl_ms": avg("inter_token_latency"),
    "median_itl_ms": pct("inter_token_latency", "p50"),
    "p99_itl_ms": pct("inter_token_latency", "p99"),
    "mean_e2el_ms": avg("request_latency"),
    "median_e2el_ms": pct("request_latency", "p50"),
    "p99_e2el_ms": pct("request_latency", "p99"),
    "input_throughput": avg("input_token_throughput"),
    "output_throughput": avg("output_token_throughput"),
    "total_token_throughput": avg("total_token_throughput"),
    "successful_requests": avg("request_count"),
    "completed": avg("request_count"),
    "benchmark_duration_s": avg("benchmark_duration")
    or data.get("benchmark_duration_s"),
    "total_input_tokens": avg("total_usage_prompt_tokens"),
    "total_output_tokens": avg("total_usage_completion_tokens"),
    "cache_hit_tokens": cache_hit_tokens,
    "cache_total_tokens": cache_total_tokens,
    "cache_hit_rate": (
        round(cache_hit_tokens / cache_total_tokens, 4)
        if cache_hit_tokens is not None and cache_total_tokens
        else None
    ),
}

payload = {key: value for key, value in payload.items() if value is not None}
dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
if cache_hit_tokens is not None and cache_total_tokens:
    print(
        f"[aiperf] prefix cache hit: {cache_hit_tokens}/{cache_total_tokens} "
        f"tokens ({cache_hit_tokens / cache_total_tokens:.2%})"
    )
else:
    print(
        "[aiperf] prefix cache hit: unavailable "
        "(AIPerf profiling cache-read counters were not produced)"
    )
print(f"[aiperf] dashboard json: {dst}")
PY
}

run_aiperf_agentic_benchmark() {
  ensure_aiperf

  if is_agentic_dpa; then
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
  else
    unset AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID
  fi

  local safe_model="${MODEL_NAME//\//-}"
  local -a server_metrics_args=(--server-metrics)
  local idx
  for idx in "${!prefill_ips[@]}"; do
    server_metrics_args+=("http://${prefill_ips[$idx]}:${prefill_ports[$idx]}/metrics")
  done
  for idx in "${!decode_ips[@]}"; do
    server_metrics_args+=("http://${decode_ips[$idx]}:${decode_ports[$idx]}/metrics")
  done

  local conc
  IFS=',' read -r -a concs <<< "${CONC_LIST}"
  for conc in "${concs[@]}"; do
    conc="${conc//[[:space:]]/}"
    [[ -n "${conc}" ]] || continue
    local out_dir="${RUN_DIR}/benchmark_results/aiperf-${safe_model}-${TOPOLOGY}-c${conc}"
    local result_file="pd-${BACKEND}-${safe_model}-${TOPOLOGY}-isl${AIPERF_MAX_CONTEXT_LENGTH}-osl1024-conc${conc}-${RANDOM_RANGE_RATIO}.json"
    local aiperf_json="${out_dir}/profile_export_aiperf.json"
    local dashboard_json="${RUN_DIR}/benchmark_results/${result_file}"
    local -a unsafe_args=()
    if (( AIPERF_BENCHMARK_DURATION < 900 )) \
      || [[ "${AIPERF_UNSAFE_OVERRIDE}" == "1" || "${AIPERF_UNSAFE_OVERRIDE}" == "true" ]]; then
      unsafe_args+=(--unsafe-override)
    fi

    echo "[aiperf] ${result_file}"
    mkdir -p "${out_dir}"
    AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT="${AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT}" \
    AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT}" \
    AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES}" \
    AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT}" \
    AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT}" \
    AIPERF_UI_REALTIME_METRICS_ENABLED=true \
      "${AIPERF_VENV}/bin/aiperf" profile \
      "${unsafe_args[@]}" \
      --scenario "${AIPERF_SCENARIO}" \
      --url "http://127.0.0.1:${ROUTER_PORT}" \
      --endpoint /v1/chat/completions \
      --endpoint-type chat \
      --streaming \
      --model "${MODEL_PATH}" \
      --concurrency "${conc}" \
      --benchmark-duration "${AIPERF_BENCHMARK_DURATION}" \
      --stats-interval 30 \
      --random-seed 42 \
      --failed-request-threshold "${AIPERF_FAILED_REQUEST_THRESHOLD}" \
      --trajectory-start-min-ratio "${AIPERF_TRAJECTORY_START_MIN_RATIO}" \
      --trajectory-start-max-ratio "${AIPERF_TRAJECTORY_START_MAX_RATIO}" \
      --warmup-requests-per-lane "${AIPERF_WARMUP_REQUESTS_PER_LANE}" \
      --trace-idle-gap-cap-seconds "${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS}" \
      --warmup-grace-period "${AIPERF_WARMUP_GRACE_PERIOD}" \
      --use-server-token-count \
      --no-gpu-telemetry \
      --tokenizer "${MODEL_PATH}" \
      --tokenizer-trust-remote-code \
      --max-context-length "${AIPERF_MAX_CONTEXT_LENGTH}" \
      --num-dataset-entries "${AIPERF_NUM_DATASET_ENTRIES}" \
      --slice-duration "${AIPERF_SLICE_DURATION}" \
      "${server_metrics_args[@]}" \
      --output-artifact-dir "${out_dir}" \
      --public-dataset "${AIPERF_PUBLIC_DATASET}" \
      2>&1 | tee "${out_dir}/aiperf.log"

    if [[ ! -f "${aiperf_json}" ]]; then
      echo "[aiperf][FAIL] ${aiperf_json} was not produced" >&2
      return 1
    fi
    write_aiperf_dashboard_json "${aiperf_json}" "${dashboard_json}" "${conc}"
  done
}

run_swebench_lite_eval() {
  local -a eval_concs=()
  local candidate
  IFS=',' read -r -a candidates <<< "${EVAL_CONCURRENCY}"
  for candidate in "${candidates[@]}"; do
    candidate="${candidate//[[:space:]]/}"
    [[ -n "${candidate}" ]] && eval_concs+=("${candidate}")
  done
  if [[ "${#eval_concs[@]}" -ne 1 || ! "${eval_concs[0]}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SWE-bench Lite requires exactly one positive eval concurrency" >&2
    return 2
  fi

  local eval_conc="${eval_concs[0]}"
  local agent_workers="${SWEBENCH_AGENT_WORKERS}"
  local tag result_dir result_file runner
  tag="$(date +%Y%m%d%H%M%S)_swebench_lite_${TOPOLOGY}_c${eval_conc}"
  result_dir="${RUN_DIR}/eval_results/${tag}"
  result_file="${result_dir}/results_swebench_lite.json"
  runner="${ATOMESH_SCRIPT_DIR}/run_swebench_lite.sh"
  mkdir -p "${result_dir}"

  if [[ ! -f "${runner}" ]]; then
    echo "ERROR: ATOM SWE-bench runner is missing: ${runner}" >&2
    return 1
  fi

  echo ""
  echo "========================================="
  echo "[eval] SWE-bench Lite local-Docker evaluation"
  echo "[eval] workers=${agent_workers} limit=${EVAL_LIMIT:-full}"
  echo "[eval] mini-swe-agent=2.4.5 swebench=4.1.0"
  echo "========================================="

  EVAL_LIMIT="${EVAL_LIMIT}" \
  SWEBENCH_AGENT_STEP_LIMIT="${SWEBENCH_AGENT_STEP_LIMIT}" \
  SWEBENCH_CASE_TIMEOUT="${SWEBENCH_CASE_TIMEOUT}" \
  SWEBENCH_AGENT_TIMEOUT="${SWEBENCH_AGENT_TIMEOUT}" \
  SWEBENCH_SCORE_TIMEOUT="${SWEBENCH_SCORE_TIMEOUT}" \
  SWEBENCH_MAX_WORKERS="${SWEBENCH_MAX_WORKERS}" \
  SWEBENCH_EVAL_TIMEOUT="${SWEBENCH_EVAL_TIMEOUT}" \
  SWEBENCH_MIN_DISK_GB="${SWEBENCH_MIN_DISK_GB}" \
  SWEBENCH_PRUNE_IMAGES="${SWEBENCH_PRUNE_IMAGES}" \
    bash "${runner}" \
      --output-dir "${result_dir}" \
      --model-name "${MODEL_NAME}" \
      --api-model "${MODEL_PATH}" \
      --api-base "http://127.0.0.1:${ROUTER_PORT}/v1" \
      --run-id "${tag}" \
      --limit "${EVAL_LIMIT:-full}" \
      --venv "${SWEBENCH_VENV}" \
      --agent-workers "${agent_workers}"

  if [[ ! -s "${result_file}" ]]; then
    echo "ERROR: SWE-bench Lite did not produce ${result_file}" >&2
    return 1
  fi

  # Trajectories are useful while the job is live but too large for the
  # benchmark artifact. Keep predictions, the official report, and score JSON.
  find "${result_dir}" -type f -name '*.traj*' -delete 2>/dev/null || true

  local score resolved total
  read -r score resolved total < <(
    python3 - "${result_file}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
task = data.get("results", {}).get("swebench_lite", {})
score = task.get("exact_match,resolved")
details = data.get("swebench", {})
resolved = details.get("resolved")
total = details.get("total")
if score is None or resolved is None or total is None:
    raise SystemExit("SWE-bench Lite result is missing score details")
print(score, resolved, total)
PY
  )
  echo "[eval] SWE-bench Lite resolved ${resolved}/${total} = ${score}"

  if [[ -n "${EVAL_THRESHOLD}" ]]; then
    python3 - "${score}" "${EVAL_THRESHOLD}" <<'PY'
import sys

score = float(sys.argv[1])
threshold = float(sys.argv[2])
if score < threshold:
    raise SystemExit(
        f"SWE-bench Lite score {score:.4f} is below threshold {threshold:.4f}"
    )
print(f"[eval] SWE-bench Lite threshold passed: {score:.4f} >= {threshold:.4f}")
PY
  fi
}

run_eval() {
  [[ "${RUN_EVAL}" == "true" ]] || [[ "${RUN_EVAL}" == "1" ]] || return 0
  if [[ "${EVAL_TASK}" == "swebench_lite" ]]; then
    run_swebench_lite_eval
    return
  fi
  if [[ "${EVAL_TASK}" != "gsm8k" ]]; then
    echo "[eval] unsupported task ${EVAL_TASK}; skipping"
    return 0
  fi
  if ! command -v lm_eval >/dev/null 2>&1; then
    python3 -m pip install 'lm-eval[api]'
  fi
  local limit_arg=()
  if [[ -n "${EVAL_LIMIT}" ]]; then
    limit_arg=(--limit "${EVAL_LIMIT}")
  fi
  local eval_extra_args=()
  if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
    eval_extra_args+=(--batch_size "${EVAL_BATCH_SIZE}")
  fi
  if [[ "${EVAL_APPLY_CHAT_TEMPLATE}" == "true" || "${EVAL_APPLY_CHAT_TEMPLATE}" == "1" ]]; then
    eval_extra_args+=(--apply_chat_template)
  fi
  if [[ "${EVAL_FEWSHOT_AS_MULTITURN}" == "true" || "${EVAL_FEWSHOT_AS_MULTITURN}" == "1" ]]; then
    eval_extra_args+=(--fewshot_as_multiturn)
  fi
  local eval_model_args_extra=""
  if [[ -n "${EVAL_MAX_GEN_TOKS}" ]]; then
    eval_model_args_extra=",max_gen_toks=${EVAL_MAX_GEN_TOKS}"
  fi
  local eval_model_args_base
  if [[ "${EVAL_MODEL_TYPE}" == "local-chat-completions" ]]; then
    eval_model_args_base="model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/${EVAL_ENDPOINT},num_concurrent="
  else
    eval_model_args_base="model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/${EVAL_ENDPOINT},num_concurrent="
    eval_model_args_extra="${eval_model_args_extra},tokenized_requests=False,trust_remote_code=True"
  fi

  IFS=',' read -r -a eval_concs <<< "${EVAL_CONCURRENCY}"
  local eval_conc tag result_dir
  for eval_conc in "${eval_concs[@]}"; do
    eval_conc="${eval_conc//[[:space:]]/}"
    [[ -n "${eval_conc}" ]] || continue
    tag="$(date +%Y%m%d%H%M%S)_gsm8k_${TOPOLOGY}_c${eval_conc}"
    result_dir="${RUN_DIR}/eval_results/${tag}"

    echo ""
    echo "========================================="
    echo "[eval] gsm8k concurrent=${eval_conc}"
    echo "========================================="

    lm_eval --model "${EVAL_MODEL_TYPE}" \
      --model_args "${eval_model_args_base}${eval_conc},max_retries=3${eval_model_args_extra}" \
      --tasks gsm8k \
      --num_fewshot "${EVAL_FEWSHOT}" \
      "${limit_arg[@]}" \
      "${eval_extra_args[@]}" \
      --output_path "${result_dir}"

    python3 - "${result_dir}" "${eval_conc}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
eval_conc = sys.argv[2]
json_files = list(result_dir.rglob("*.json")) if result_dir.is_dir() else []
if not json_files:
    print("[eval] ERROR: no result JSON found")
    raise SystemExit(1)

result_file = max(json_files, key=lambda path: path.stat().st_mtime)
data = json.loads(result_file.read_text(encoding="utf-8"))
score = (
    data.get("results", {})
    .get("gsm8k", {})
    .get("exact_match,flexible-extract", "N/A")
)
print("=========================================")
print(f"[eval] concurrent={eval_conc} exact_match,flexible-extract = {score}")
print("=========================================")
print(json.dumps(data.get("results", {}), indent=2))
PY
  done

  echo "[eval] gsm8k runs done, results saved to ${RUN_DIR}/eval_results"
}

run_benchmark_and_eval() {
  if [[ "${ATOMESH_EXECUTION_PHASE}" == "benchmark" ]]; then
    run_benchmark
    return
  fi
  if [[ "${ATOMESH_EXECUTION_PHASE}" == "eval" ]]; then
    run_eval
    return
  fi
  if [[ "${BENCHMARK_KIND}" == "aiperf_agentic" \
    && "${EVAL_TASK}" == "swebench_lite" \
    && ( "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1" ) ]]; then
    # Agentic performance cases require a fresh prefix-cache state. Run their
    # trace benchmark before the independent SWE-bench workload.
    run_benchmark
    run_eval
  else
    run_eval
    run_benchmark
  fi
}

write_metadata

if [[ "${NODE_RANK}" -eq 0 && "${SINGLE_NODE_PD}" == "1" ]]; then
  start_prefill "prefill-rank-0"
  prefill_pid="${server_pid}"
  decode_handshake_port=$((HANDSHAKE_PORT + PREFILL_TP_SIZE))
  start_decode "decode-rank-0" "${DECODE_PORT}" "${decode_handshake_port}"
  decode_pid="${server_pid}"
  trap 'cleanup_processes ${router_pid:-} ${prefill_pid:-} ${decode_pid:-}' EXIT
  for ip in "${prefill_ips[@]}"; do
    wait_http "http://${ip}:${PREFILL_PORT}/health" "prefill-${ip}" "${WAIT_SERVER_TIMEOUT}" "${prefill_pid}"
  done
  for ip in "${decode_ips[@]}"; do
    wait_http "http://${ip}:${DECODE_PORT}/health" "decode-${ip}" "${WAIT_SERVER_TIMEOUT}" "${decode_pid}"
  done
  start_router
  wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "router" "${WAIT_ROUTER_TIMEOUT}"
  run_benchmark_and_eval
  cleanup_processes "${router_pid}" "${prefill_pid}" "${decode_pid}"
elif [[ "${NODE_RANK}" -eq 0 && "${PREFILL_SINGLE_NODE_PD}" == "1" ]]; then
  prefill_pids=()
  for idx in $(seq 0 $((xP - 1))); do
    gpu_start=$((idx * PREFILL_TP_SIZE))
    gpu_end=$((gpu_start + PREFILL_TP_SIZE - 1))
    export HIP_VISIBLE_DEVICES="$(seq -s, "${gpu_start}" "${gpu_end}")"
    prefill_port="${prefill_ports[$idx]}"
    handshake_port=$((HANDSHAKE_PORT + idx * PREFILL_TP_SIZE))
    prefill_dp_master_port=$((PREFILL_DP_MASTER_PORT + idx * 200))
    prefill_dp_base_port=$((PREFILL_DP_BASE_PORT + idx * 200))
    start_prefill "prefill-rank-0-worker-${idx}" "${prefill_port}" "${handshake_port}" "${prefill_dp_master_port}" "${prefill_dp_base_port}"
    prefill_pids+=("${server_pid}")
  done
  trap 'cleanup_processes ${router_pid:-} ${prefill_pids[*]:-}' EXIT
  for idx in "${!prefill_ips[@]}"; do
    wait_http "http://${prefill_ips[$idx]}:${prefill_ports[$idx]}/health" \
      "prefill-${prefill_ips[$idx]}:${prefill_ports[$idx]}" \
      "${WAIT_SERVER_TIMEOUT}" "${prefill_pids[$idx]}"
  done
  for idx in "${!decode_ips[@]}"; do
    wait_http "http://${decode_ips[$idx]}:${decode_ports[$idx]}/health" \
      "decode-${decode_ips[$idx]}:${decode_ports[$idx]}" \
      "${WAIT_SERVER_TIMEOUT}"
  done
  start_router
  wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "router" "${WAIT_ROUTER_TIMEOUT}"
  run_benchmark_and_eval
  cleanup_processes "${router_pid}" "${prefill_pids[@]}"
elif [[ "${NODE_RANK}" -eq 0 ]]; then
  start_prefill "prefill-rank-0"
  trap 'cleanup_processes ${router_pid:-} ${server_pid:-}' EXIT
  for idx in "${!prefill_ips[@]}"; do
    wait_http "http://${prefill_ips[$idx]}:${prefill_ports[$idx]}/health" \
      "prefill-${prefill_ips[$idx]}:${prefill_ports[$idx]}" \
      "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  done
  for idx in "${!decode_ips[@]}"; do
    wait_http "http://${decode_ips[$idx]}:${decode_ports[$idx]}/health" \
      "decode-${decode_ips[$idx]}:${decode_ports[$idx]}" \
      "${WAIT_SERVER_TIMEOUT}"
  done
  start_router
  wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "router" "${WAIT_ROUTER_TIMEOUT}"
  run_benchmark_and_eval
  kill "${router_pid}" "${server_pid}" 2>/dev/null || true
elif [[ "${DECODE_SINGLE_NODE_PD}" == "1" && "${NODE_RANK}" -eq "${xP}" ]]; then
  decode_pids=()
  for idx in $(seq 0 $((yD - 1))); do
    gpu_start=$((idx * DECODE_TP_SIZE))
    gpu_end=$((gpu_start + DECODE_TP_SIZE - 1))
    export HIP_VISIBLE_DEVICES="$(seq -s, "${gpu_start}" "${gpu_end}")"
    decode_port="${decode_ports[$idx]}"
    decode_handshake_port=$((HANDSHAKE_PORT + idx * DECODE_TP_SIZE))
    decode_dp_master_port=$((DECODE_DP_MASTER_PORT + idx * 200))
    decode_dp_base_port=$((DECODE_DP_BASE_PORT + idx * 200))
    start_decode "decode-rank-${NODE_RANK}-worker-${idx}" "${decode_port}" "${decode_handshake_port}" "${decode_dp_master_port}" "${decode_dp_base_port}"
    decode_pids+=("${server_pid}")
  done
  trap 'cleanup_processes ${decode_pids[*]:-}' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}"
  wait_router_closed
  cleanup_processes "${decode_pids[@]}"
elif [[ "${PREFILL_SINGLE_NODE_PD}" == "1" ]]; then
  start_decode
  trap 'cleanup_processes ${server_pid:-}' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  cleanup_processes "${server_pid}"
elif [[ "${NODE_RANK}" -lt "${xP}" ]]; then
  start_prefill "prefill-rank-${NODE_RANK}"
  trap 'cleanup_processes ${server_pid:-}' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  cleanup_processes "${server_pid}"
else
  start_decode
  trap 'cleanup_processes ${server_pid:-}' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  cleanup_processes "${server_pid}"
fi
