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
PACKED_NODES_PD=0
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
  packed_nodes)
    PACKED_NODES_PD=1
    ;;
esac

xP="${xP:-1}"
yD="${yD:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
PREFILL_DCP_SIZE="${PREFILL_DCP_SIZE:-1}"
DECODE_DCP_SIZE="${DECODE_DCP_SIZE:-1}"
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
# Mooncake Store (MOONCAKE_STORE=1): master RPC, its HTTP metadata and metrics
# servers, and the owners' service ports (owner i uses the base + i). Below
# the ephemeral port range (32768+), where an outgoing connection of this job
# or another could hold one when a master or owner binds it.
ATOMESH_MOONCAKE_MASTER_PORT="${ATOMESH_MOONCAKE_MASTER_PORT:-26051}"
ATOMESH_MOONCAKE_METADATA_PORT="${ATOMESH_MOONCAKE_METADATA_PORT:-26080}"
ATOMESH_MOONCAKE_METRICS_PORT="${ATOMESH_MOONCAKE_METRICS_PORT:-26090}"
ATOMESH_MOONCAKE_OWNER_PORT="${ATOMESH_MOONCAKE_OWNER_PORT:-26052}"
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
ATOMESH_MOONCAKE_MASTER_PORT=$((ATOMESH_MOONCAKE_MASTER_PORT + ATOMESH_SERVICE_PORT_OFFSET))
ATOMESH_MOONCAKE_METADATA_PORT=$((ATOMESH_MOONCAKE_METADATA_PORT + ATOMESH_SERVICE_PORT_OFFSET))
ATOMESH_MOONCAKE_METRICS_PORT=$((ATOMESH_MOONCAKE_METRICS_PORT + ATOMESH_SERVICE_PORT_OFFSET))
ATOMESH_MOONCAKE_OWNER_PORT=$((ATOMESH_MOONCAKE_OWNER_PORT + ATOMESH_SERVICE_PORT_OFFSET))
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
# The Store's ports only bound the offset when it runs (mooncake_store_requested).
if [[ "${ATOMESH_PREFILL_ENV_MOONCAKE_STORE:-${MOONCAKE_STORE:-0}}" == "1" ]]; then
  for shifted_port_name in \
    ATOMESH_MOONCAKE_MASTER_PORT \
    ATOMESH_MOONCAKE_METADATA_PORT \
    ATOMESH_MOONCAKE_METRICS_PORT \
    ATOMESH_MOONCAKE_OWNER_PORT; do
    validate_shifted_port "${shifted_port_name}"
  done
fi
unset shifted_port_name
unset -f validate_shifted_port
USE_EXPLICIT_DP_PORTS=0
if [[ "${SINGLE_NODE_PD}" == "1" || "${PREFILL_SINGLE_NODE_PD}" == "1" || "${DECODE_SINGLE_NODE_PD}" == "1" || "${PACKED_NODES_PD}" == "1" ]]; then
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
SPEC_DECODE_ACCEPTANCE_LENGTH="${SPEC_DECODE_ACCEPTANCE_LENGTH:-}"
STATE_CHECKPOINT_INTERVAL_TOKENS="${STATE_CHECKPOINT_INTERVAL_TOKENS:-}"
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
AIPERF_APPLY_CHAT_TEMPLATE="${AIPERF_APPLY_CHAT_TEMPLATE:-false}"
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
  env | grep -E '^(HIP_|HSA_|AITER_|ATOM_|RCCL_|NCCL_|CUDA_|MOONCAKE_|MORI_|UCX_)' | sort || true
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

# GPU timing is opt-in; agentic reports need it on both service roles.
if [[ "${BENCHMARK_KIND}" == "aiperf_agentic" ]]; then
  export ATOMESH_PREFILL_ENV_ATOM_ENABLE_METRICS_DEVICE_TIMER="${ATOMESH_PREFILL_ENV_ATOM_ENABLE_METRICS_DEVICE_TIMER:-${ATOM_ENABLE_METRICS_DEVICE_TIMER:-1}}"
  export ATOMESH_DECODE_ENV_ATOM_ENABLE_METRICS_DEVICE_TIMER="${ATOMESH_DECODE_ENV_ATOM_ENABLE_METRICS_DEVICE_TIMER:-${ATOM_ENABLE_METRICS_DEVICE_TIMER:-1}}"
fi

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
elif [[ "${PACKED_NODES_PD}" == "1" ]]; then
  # PP size is not a top-level env var on this launcher; parse it from extra_args.
  _pp_from_args() {
    local args="$1"
    if [[ "${args}" =~ --pipeline-parallel-size[=\ ]+([0-9]+) ]]; then
      echo "${BASH_REMATCH[1]}"
    else
      echo 1
    fi
  }
  PREFILL_PP_SIZE="$(_pp_from_args "${PREFILL_SERVER_ARGS}")"
  DECODE_PP_SIZE="$(_pp_from_args "${DECODE_SERVER_ARGS}")"
  prefill_nodes=()
  prefill_gpus=()
  decode_nodes=()
  decode_gpus=()
  packed_node=0
  packed_used=0
  place_packed_worker() {
    local width="$1"
    local -n placed_nodes="$2"
    local -n placed_gpus="$3"
    if (( width > 8 )); then
      echo "ERROR: packed_nodes workers must fit on one 8-GPU node" >&2
      exit 2
    fi
    if (( packed_used + width > 8 )); then
      packed_node=$((packed_node + 1))
      packed_used=0
    fi
    placed_nodes+=("${packed_node}")
    placed_gpus+=("$(seq -s, "${packed_used}" "$((packed_used + width - 1))")")
    packed_used=$((packed_used + width))
  }
  for idx in $(seq 0 $((xP - 1))); do
    place_packed_worker "$((PREFILL_PP_SIZE * PREFILL_TP_SIZE))" prefill_nodes prefill_gpus
    prefill_ips+=("${IP_ARRAY[${prefill_nodes[$idx]}]:-}")
    prefill_ports+=("$((PREFILL_PORT + idx))")
    prefill_args+=(--prefill "http://${prefill_ips[$idx]}:${prefill_ports[$idx]}")
  done
  for idx in $(seq 0 $((yD - 1))); do
    place_packed_worker "$((DECODE_PP_SIZE * DECODE_TP_SIZE))" decode_nodes decode_gpus
    decode_ips+=("${IP_ARRAY[${decode_nodes[$idx]}]:-}")
    decode_ports+=("$((DECODE_PORT + idx))")
    decode_args+=(--decode "http://${decode_ips[$idx]}:${decode_ports[$idx]}")
  done
  if (( packed_node + 1 != ${#IP_ARRAY[@]} )); then
    echo "ERROR: packed_nodes needs $((packed_node + 1)) node(s), got ${#IP_ARRAY[@]}" >&2
    exit 2
  fi
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

prefill_parallel=(
  -tp "${PREFILL_TP_SIZE}"
  --decode-context-parallel-size "${PREFILL_DCP_SIZE}"
)
if [[ "${PREFILL_ENABLE_DP}" == "true" ]]; then
  prefill_parallel+=("--enable-dp-attention")
fi

decode_parallel=(
  -tp "${DECODE_TP_SIZE}"
  --decode-context-parallel-size "${DECODE_DCP_SIZE}"
)
if [[ "${DECODE_ENABLE_DP}" == "true" ]]; then
  decode_parallel+=("--enable-dp-attention")
fi

# AgentX captures every query-token count the engine can produce, i.e. the dense
# range [2, graph_max] with graph_max = seqs * (1 + spec_tokens), where seqs
# defaults to 2 * CONC. Concurrencies whose in-flight window is wider than
# 2 * CONC pin seqs explicitly via cudagraph_max_num_seqs.
auto_cudagraph_capture_sizes() {
  local role="$1"
  local seqs="$2"
  local conc spec graph_max
  spec="${NUM_SPEC_TOKENS:-0}"
  [[ "${spec}" =~ ^[0-9]+$ ]] || spec=0
  if [[ ! "${seqs}" =~ ^[0-9]+$ ]]; then
    conc="$(echo "${BENCH_MAX_CONCURRENCY}" | tr 'x,' '\n' | sort -n | tail -1)"
    [[ "${conc}" =~ ^[0-9]+$ ]] || conc=1
    seqs=$(( 2 * conc ))
  fi
  graph_max=$(( seqs * (1 + spec) ))
  if (( graph_max < 2 )); then
    graph_max=2
  fi
  echo "[${role}] cudagraph auto range 2..${graph_max} (seqs=${seqs} spec=${spec})" >&2
  echo "[$(seq -s, 2 "${graph_max}")]"
}

build_cudagraph_args() {
  local role="$1"
  local -n out="$2"
  local prefix="${role^^}"
  local sizes_var="${prefix}_CUDAGRAPH"
  local mode_var="${prefix}_CUDAGRAPH_MODE"
  local level_var="${prefix}_COMPILATION_LEVEL"
  local seqs_var="${prefix}_CUDAGRAPH_MAX_NUM_SEQS"
  local mode="${!mode_var:-}"
  local level="${!level_var:-}"
  case "${!sizes_var:-}" in
    ""|none|None|NONE|false|False|FALSE|off|Off|OFF|disabled|Disabled|DISABLED)
      out=()
      ;;
    auto|Auto|AUTO)
      out=(
        --cudagraph-capture-sizes
        "$(auto_cudagraph_capture_sizes "${role}" "${!seqs_var:-}")"
      )
      ;;
    *)
      out=(--cudagraph-capture-sizes "${!sizes_var}")
      ;;
  esac
  if [[ -n "${mode}" ]]; then
    out+=(--cudagraph-mode "${mode}")
  fi
  if [[ -n "${level}" ]]; then
    out+=(--level "${level}")
  fi
}

prefill_cudagraph_args=()
decode_cudagraph_args=()
build_cudagraph_args prefill prefill_cudagraph_args
build_cudagraph_args decode decode_cudagraph_args

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
spec_decode_acceptance_for_server="${SPEC_DECODE_ACCEPTANCE_LENGTH}"
if [[ "${ATOMESH_EXECUTION_PHASE}" == "eval" && "${EVAL_TASK}" == "gsm8k" ]]; then
  spec_decode_acceptance_for_server=""
  echo "[runtime] omitting spec-decode-acceptance-length for gsm8k eval phase"
fi
if [[ -n "${spec_decode_acceptance_for_server}" ]]; then
  server_common+=(
    --spec-decode-acceptance-length "${spec_decode_acceptance_for_server}"
  )
fi
if [[ -n "${STATE_CHECKPOINT_INTERVAL_TOKENS}" ]]; then
  server_common+=(
    --state-checkpoint-interval-tokens "${STATE_CHECKPOINT_INTERVAL_TOKENS}"
  )
fi
wait_http() {
  local url="$1"
  local name="$2"
  local timeout="$3"
  local pid="${4:-}"
  local deadline=$(( $(date +%s) + timeout ))
  echo "[wait] ${name} ${url} timeout=${timeout}s"
  until curl -sf --max-time 10 "${url}" >/dev/null 2>&1; do
    exit_if_mooncake_store_died
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
      exit_if_mooncake_store_died
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

  if [[ "${PACKED_NODES_PD:-0}" == "1" ]]; then
    echo "[runtime] logging ${log_file} (file-only, packed layout)"
    if command -v setsid >/dev/null 2>&1; then
      setsid "$@" >"${log_file}" 2>&1 &
    else
      "$@" >"${log_file}" 2>&1 &
    fi
  elif command -v setsid >/dev/null 2>&1; then
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
    # Not a bare return: in a function the EXIT trap runs, that returns the
    # trap's $? (the exit status), and cleanup would then wait on live servers.
    return $?
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
    # A process in uninterruptible sleep (a GPU or RDMA teardown) outlives
    # SIGKILL, and `wait` would block on it until the job's time limit.
    deadline=$(( $(date +%s) + 30 ))
    while process_is_running "${pid}" && [[ "$(date +%s)" -lt "${deadline}" ]]; do
      sleep 1
    done
    if process_is_running "${pid}"; then
      echo "[cleanup] WARNING: pid ${pid} is still running 30s after SIGKILL (state $(awk '{ print $3 }' "/proc/${pid}/stat" 2>/dev/null || echo '?')); not waiting for it" >&2
      return 0
    fi
  fi
  wait "${pid}" 2>/dev/null || true
}

# LMCache's disk tier lives on a host bind mount, so unlike the container's own
# /tmp it survives `docker run --rm`. Every concurrency runs as its own job, and
# a tier left behind would both serve the previous job's KV and hold its
# LMCACHE_MAX_LOCAL_DISK_SIZE of disk per rank. Start empty, leave nothing.
lmcache_disk_dir=""

reset_lmcache_disk() {
  local dir="${LMCACHE_LOCAL_DISK:-}"
  [[ -n "${dir}" && "${dir}" != "/" ]] || return 0
  # The repository working directory is read-only in the container. Resolve
  # relative paths under the writable logs, isolated by job, phase and node rank.
  if [[ "${dir}" != /* ]]; then
    dir="${RUNTIME_LOG_DIR}/rank-${NODE_RANK}/${dir#./}"
  fi
  export LMCACHE_LOCAL_DISK="${dir}"
  # Several prefill workers can share this shell, so only the first one empties
  # the tier; a later one would delete a running worker's cache underneath it.
  [[ "${lmcache_disk_dir}" != "${dir}" ]] || return 0
  lmcache_disk_dir="${dir}"
  rm -rf -- "${dir}"
  mkdir -p -- "${dir}"
  echo "[lmcache] disk tier ${dir} reset (${LMCACHE_MAX_LOCAL_DISK_SIZE:-0}GiB per rank)"
}

purge_lmcache_disk() {
  [[ -n "${lmcache_disk_dir}" ]] || return 0
  rm -rf -- "${lmcache_disk_dir}"
  echo "[lmcache] disk tier ${lmcache_disk_dir} removed"
  lmcache_disk_dir=""
}

# MOONCAKE_STORE=1 (prefill role env) gives this shell's prefill workers a
# Mooncake Store as their KV offload tier: ATOM's mooncake_store offload
# connector saves each PP stage's full chunks of a prompt to the Store, whose
# memory belongs to owner processes started here, and loads a prefix hit back
# over RDMA, with no CPU tier in the workers.
#  - A master, on the CPUs of MOONCAKE_STORE_MASTER_NUMA (default 1), and one
#    owner per MOONCAKE_STORE_OWNERS entry "<numa>:<GiB>[:<rdma,...>]"
#    (default "0:768:rdma4,rdma5;1:768:rdma6,rdma7"), bound to that node by
#    numa_exec.py. An owner without its own device list serves on
#    MOONCAKE_STORE_OWNER_RDMA_DEVICES (default rdma4-rdma7). ATOM gives each
#    prefill stage one NIC of its own, the GPU's, and under one shared master
#    refuses one the owners use: owners and requesters sharing NICs there
#    stall concurrent reads. The default owners mount on pit2-p03 (TW MI355X)
#    nodes: about 1.5 TiB per NUMA node, rdma4-7 on NUMA1, and an ionic NIC
#    that registers at most 832-896 GiB for one process -- a 960 GiB owner
#    fails to mount on one NIC or four, and two owners of 768 and 960 GiB on
#    the same four NICs failed as well, while 768 + 768 GiB on separate NIC
#    pairs mount in 12 s.
#  - MOONCAKE_STORE_POOLS=per_nic (default shared) is for nodes whose owners
#    must share the stages' NICs, e.g. a job that may use only the memory and
#    NICs of its own GPUs: one master per owner NIC, every owner on exactly one
#    NIC, and ATOM points the stage on a NIC at that NIC's pool
#    (mooncake_store.pools). A NIC then carries one stage's reads from its own
#    pool's owners, not every owner's. Across two nodes, four stages read
#    147 GB/s this way with no retransmission.
#  - MOONCAKE_STORE_DECODE_OWNERS, in the same format, are owners the decode
#    node starts on its own memory (pd_worker_layout multi_node, one prefill
#    node and one decode node), joining the prefill node's masters. The
#    prefill workers start once every pool counts the owners of both nodes.
#  - Every Mooncake process runs with MC_NUM_QP_PER_EP=1: the master, the
#    owners, the prefill workers and the decode workers, whose P->D transfer
#    engine reads the same variable (peers with different QP counts cannot
#    connect). The decode gets it on every node of the job.
#  - The launcher owns the workers' ATOM_KV_OFFLOAD and
#    ATOM_KV_OFFLOAD_EXTRA_CONFIG here, so PREFILL_KV_TRANSFER_CONFIG names the
#    P/D connector only. MOONCAKE_STORE_CONNECTOR_CONFIG, a JSON object of the
#    connector's other keys (mooncake_store.*, max_pending_saves), joins the
#    extra config, e.g. {"mooncake_store.pool_device":"cpu"}; an unknown key
#    or a value of the wrong type is refused before anything starts.
#  - A lookup or a read leases its objects for MOONCAKE_STORE_LEASE_TTL_MS
#    (default 10000): eviction skips them until then, and a read that ends
#    later fails.
#  - The owners pin huge pages under MPOL_BIND. Before they start, the clean
#    page cache of MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS (colon-separated,
#    default the model) is dropped, the owners' pins and
#    MOONCAKE_STORE_HOST_POOL_GIB per prefill GPU (the connector's transfer
#    pool when it sits in host memory; default 0, in GPU memory) must fit each
#    node with MOONCAKE_STORE_NODE_RESERVE_GIB (default 128) to spare, and each
#    node is compacted for its pins (numa_memory_budget.py --compact) before
#    the owners fault them.
# validate_mooncake_store_settings refuses a conflicting setting, or an image
# without the Store, before the node starts anything. Master and owners start
# before the workers, which connect (and round-trip a probe chunk) while they
# build their engines and never retry, and stop after them. A Store lives as
# long as this shell: its objects outlive a worker restart, not the job.
mooncake_store_pids=()
mooncake_store_logs=()
mooncake_store_names=()
# Exit status of the first Store failure: a process that died before it was
# stopped, or owner capacity the master lost during the run.
mooncake_store_failed_rc=""
mooncake_page_cache_dropper_pid=""
mooncake_owner_plan_numa=()
mooncake_owner_plan_gib=()
mooncake_owner_plan_devices=()
# One entry per pool, numbered alike on every node: its NIC under per_nic (""
# for the one shared pool), and the segment bytes its master counts once the
# owners of every node have mounted.
mooncake_pool_devices=()
mooncake_pool_capacity=()
# The first entries of mooncake_store_pids are this node's masters: one per
# pool on the prefill node, none on a decode node.
mooncake_store_master_count=0
# Set once every pool counts its planned capacity, and once the masters'
# metrics of the run are saved.
mooncake_store_ready=0
mooncake_store_metrics_saved=0
# The prefill workers' env for the running Store, set by start_mooncake_store.
mooncake_store_prefill_env=()
# The prefill GPUs whose host transfer pools the running Store's memory plan
# covers.
mooncake_store_started_for=""

# The prefill role env asks for it; the decode role reads the same variable
# from ATOMESH_PREFILL_ENV_ on any node.
mooncake_store_requested() {
  [[ "${ATOMESH_PREFILL_ENV_MOONCAKE_STORE:-${MOONCAKE_STORE:-0}}" == "1" ]]
}

# mooncake_setting <name> [default]: a Store setting of the prefill role env,
# read the same way on a decode node, where only ATOMESH_PREFILL_ENV_<name>
# carries it.
mooncake_setting() {
  local prefixed="ATOMESH_PREFILL_ENV_$1"
  local plain="$1"
  printf '%s' "${!prefixed:-${!plain:-${2:-}}}"
}

# mooncake_pool_port <base port> <pool>: the pool's master, metadata or
# metrics port, 100 apart so the owners' service ports stay clear.
mooncake_pool_port() {
  echo $(( $1 + 100 * $2 ))
}

# mooncake_pool_suffix <pool>: the pool's log and metrics file suffix; none
# for the one shared pool.
mooncake_pool_suffix() {
  if [[ "${#mooncake_pool_devices[@]}" -gt 1 || -n "${mooncake_pool_devices[0]:-}" ]]; then
    printf -- '-pool%s' "$1"
  fi
}

# Every Mooncake process needs one QP per endpoint; refuses any other value.
require_mooncake_qp_per_endpoint() {
  if [[ -n "${MC_NUM_QP_PER_EP:-}" && "${MC_NUM_QP_PER_EP}" != "1" ]]; then
    echo "[mooncake-store][FAIL] MC_NUM_QP_PER_EP=${MC_NUM_QP_PER_EP}: the Mooncake Store needs 1 in every Mooncake process, P->D included" >&2
    exit 2
  fi
}

# Each Store client and owner is given its NICs: a prefill worker the GPU's
# own, an owner its --device_names. MC_MS_AUTO_DISC=1 makes Mooncake take every
# NIC of the node instead, which stalls concurrent reads; the workers refuse it
# only once their model has loaded.
refuse_mooncake_nic_auto_discovery() {
  local auto_discovery="${MC_MS_AUTO_DISC:-}"
  if [[ "${auto_discovery//[[:space:]]/}" == "1" ]]; then
    echo "[mooncake-store][FAIL] MC_MS_AUTO_DISC=1 makes the Store's clients and owners use every NIC instead of the ones they are given; unset it" >&2
    exit 2
  fi
}

# parse_mooncake_owner_spec <setting> <numa[]> <gib[]> <devices[]>: appends
# the "<numa>:<GiB>[:<rdma,...>]" entries of MOONCAKE_STORE_OWNERS or
# MOONCAKE_STORE_DECODE_OWNERS to the named arrays; exits 2 on an invalid
# configuration.
parse_mooncake_owner_spec() {
  local setting="$1"
  local -n numa_out="$2"
  local -n gib_out="$3"
  local -n devices_out="$4"
  local default_spec=""
  if [[ "${setting}" == "MOONCAKE_STORE_OWNERS" ]]; then
    default_spec="0:768:rdma4,rdma5;1:768:rdma6,rdma7"
  fi
  local spec default_devices
  spec="$(mooncake_setting "${setting}" "${default_spec}")"
  default_devices="$(mooncake_setting MOONCAKE_STORE_OWNER_RDMA_DEVICES rdma4,rdma5,rdma6,rdma7)"
  local device_list='[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*'
  if [[ ! "${default_devices}" =~ ^${device_list}$ ]]; then
    echo "[mooncake-store][FAIL] MOONCAKE_STORE_OWNER_RDMA_DEVICES=${default_devices} is not a comma-separated device list" >&2
    exit 2
  fi
  local -a entries=()
  IFS=';' read -r -a entries <<< "${spec}"
  if [[ "${#entries[@]}" -eq 0 ]]; then
    echo "[mooncake-store][FAIL] ${setting} is empty" >&2
    exit 2
  fi
  local entry
  for entry in "${entries[@]}"; do
    if [[ ! "${entry}" =~ ^([0-9]+):([0-9]+)(:(${device_list}))?$ ]] \
      || (( 10#${BASH_REMATCH[2]} == 0 )); then
      echo "[mooncake-store][FAIL] ${setting} entry '${entry}' is not <numa>:<GiB>[:<rdma,...>] with GiB > 0" >&2
      exit 2
    fi
    numa_out+=("$(( 10#${BASH_REMATCH[1]} ))")
    gib_out+=("$(( 10#${BASH_REMATCH[2]} ))")
    devices_out+=("${BASH_REMATCH[4]:-${default_devices}}")
  done
}

# plan_mooncake_store_owners [setting]: fills the mooncake_owner_plan_* arrays
# with this node's owners, MOONCAKE_STORE_OWNERS on the prefill node (the
# default) or MOONCAKE_STORE_DECODE_OWNERS on a decode node.
plan_mooncake_store_owners() {
  mooncake_owner_plan_numa=()
  mooncake_owner_plan_gib=()
  mooncake_owner_plan_devices=()
  parse_mooncake_owner_spec "${1:-MOONCAKE_STORE_OWNERS}" \
    mooncake_owner_plan_numa mooncake_owner_plan_gib mooncake_owner_plan_devices
}

# Fills mooncake_pool_devices and mooncake_pool_capacity from the owners of
# every node, numbering the pools alike on each; exits 2 on an invalid setting.
plan_mooncake_store_pools() {
  local mode
  mode="$(mooncake_setting MOONCAKE_STORE_POOLS shared)"
  local -a numa=() gib=() devices=()
  parse_mooncake_owner_spec MOONCAKE_STORE_OWNERS numa gib devices
  if [[ -n "$(mooncake_setting MOONCAKE_STORE_DECODE_OWNERS)" ]]; then
    parse_mooncake_owner_spec MOONCAKE_STORE_DECODE_OWNERS numa gib devices
  fi
  mooncake_pool_devices=()
  mooncake_pool_capacity=()
  local i device bytes
  case "${mode}" in
    shared)
      bytes=0
      for i in "${!gib[@]}"; do
        bytes=$(( bytes + gib[i] * 1024 * 1024 * 1024 ))
      done
      mooncake_pool_devices=("")
      mooncake_pool_capacity=("${bytes}")
      ;;
    per_nic)
      for i in "${!devices[@]}"; do
        if [[ "${devices[i]}" == *,* ]]; then
          echo "[mooncake-store][FAIL] MOONCAKE_STORE_POOLS=per_nic gives each NIC a pool of its own, so every owner needs exactly one NIC; ${numa[i]}:${gib[i]}:${devices[i]} names several" >&2
          exit 2
        fi
      done
      while read -r device; do
        bytes=0
        for i in "${!devices[@]}"; do
          if [[ "${devices[i]}" == "${device}" ]]; then
            bytes=$(( bytes + gib[i] * 1024 * 1024 * 1024 ))
          fi
        done
        mooncake_pool_devices+=("${device}")
        mooncake_pool_capacity+=("${bytes}")
      done < <(printf '%s\n' "${devices[@]}" | sort -u -V)
      ;;
    *)
      echo "[mooncake-store][FAIL] MOONCAKE_STORE_POOLS=${mode} is neither shared nor per_nic" >&2
      exit 2
      ;;
  esac
  local base port last=$(( ${#mooncake_pool_devices[@]} - 1 ))
  for base in ATOMESH_MOONCAKE_MASTER_PORT ATOMESH_MOONCAKE_METADATA_PORT ATOMESH_MOONCAKE_METRICS_PORT; do
    port="$(mooncake_pool_port "${!base}" "${last}")"
    if (( port > 65535 )); then
      echo "[mooncake-store][FAIL] pool ${last}'s port ${base} + 100 x ${last} = ${port} is outside the valid TCP port range" >&2
      exit 2
    fi
  done
}

# The owners' devices, each once, in plan order.
mooncake_owner_devices_csv() {
  local devices csv="" device
  for devices in "${mooncake_owner_plan_devices[@]}"; do
    for device in ${devices//,/ }; do
      [[ ",${csv}," == *",${device},"* ]] || csv+="${csv:+,}${device}"
    done
  done
  printf '%s' "${csv}"
}

# kv_transfer_offload_connector <kv-transfer-config JSON>: prints the first
# offload connector the config names, matched by name or alias as ATOM matches
# them (atom/model_engine/arg_utils.py, _OFFLOAD_CONNECTORS: any case, outer
# whitespace ignored); fails when it names none.
kv_transfer_offload_connector() {
  local config="${1,,}" name
  config="${config//[[:space:]]/}"
  for name in lmcache_offload lmcacheoffloadconnector lmcacheconnectorv1 \
    lmcache_mp lmcachempconnector mooncake_store mooncakestoreoffloadconnector; do
    if [[ "${config}" == *"\"kv_connector\":\"${name}\""* ]]; then
      printf '%s' "${name}"
      return 0
    fi
  done
  return 1
}

# mooncake_store_worker_config [launcher JSON object]: that object with the
# case's MOONCAKE_STORE_CONNECTOR_CONFIG merged in, the prefill workers'
# ATOM_KV_OFFLOAD_EXTRA_CONFIG; exits 2 when the case's value is not a JSON
# object of the connector's settings, each of the type the connector takes,
# or sets one of the keys the launcher derives from the Store it starts. The
# workers would refuse such a value too, but only after the Store started.
mooncake_store_worker_config() {
  python3 -c '
import json
import math
import sys

config = json.loads(sys.argv[1] or "{}")
text = sys.argv[2].strip()
launcher_keys = {"mooncake_store." + key for key in sys.argv[3].split()}
name = "MOONCAKE_STORE_CONNECTOR_CONFIG"


def positive_integer(value):
    return type(value) is int and value > 0


def positive_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


# The settings a case may give the connector, as it parses them
# (atom/kv_transfer/offload/mooncake_store/config.py, whose _FIELDS are these
# and the launcher keys; tests/test_atomesh_mooncake_store.py keeps them equal).
settings = {
    "mooncake_store.rdma_devices": (
        "a comma-separated string",
        lambda value: isinstance(value, str),
    ),
    "mooncake_store.chunk_tokens": ("a positive integer", positive_integer),
    "mooncake_store.pool_device": (
        "\"gpu\" or \"cpu\"",
        lambda value: value in ("gpu", "cpu"),
    ),
    "mooncake_store.load_pool_mib": ("a positive integer", positive_integer),
    "mooncake_store.save_pool_mib": ("a positive integer", positive_integer),
    "mooncake_store.lookup_batch_keys": ("a positive integer", positive_integer),
    "mooncake_store.save_abandon_timeout_s": (
        "a positive number of seconds",
        positive_seconds,
    ),
    "mooncake_store.publish_loaded_prefix": (
        "true or false",
        lambda value: isinstance(value, bool),
    ),
    "mooncake_store.startup_probe": (
        "true or false",
        lambda value: isinstance(value, bool),
    ),
    "mooncake_store.direct_copy": (
        "true or false",
        lambda value: isinstance(value, bool),
    ),
    "max_pending_saves": ("a positive integer", positive_integer),
}
try:
    extra = json.loads(text) if text else {}
except ValueError as exc:
    sys.exit(f"[mooncake-store][FAIL] {name} is not JSON: {exc}")
if not isinstance(extra, dict):
    sys.exit(f"[mooncake-store][FAIL] {name} must be a JSON object")
for key, value in extra.items():
    if key in launcher_keys:
        sys.exit(
            f"[mooncake-store][FAIL] {name} sets {key}, which the launcher "
            "sets from the Store it starts"
        )
    if key not in settings:
        known = ", ".join(settings)
        sys.exit(
            f"[mooncake-store][FAIL] {name} sets {key}: it takes the "
            f"mooncake_store connector settings {known} only"
        )
    kind, valid = settings[key]
    if not valid(value):
        sys.exit(
            f"[mooncake-store][FAIL] {name} sets {key} to {json.dumps(value)}: "
            f"it must be {kind}"
        )
print(json.dumps({**config, **extra}, separators=(",", ":")))
' "${1:-}" "${MOONCAKE_STORE_CONNECTOR_CONFIG:-}" \
    "master metadata pools owner_rdma_devices local_hostname protocol" \
    || exit 2
}

# Refuses a prefill setting the Store cannot run with, and fills the owner
# plan. Independent of the node: the owners' devices and memory are checked
# where the Store starts.
check_mooncake_store_settings() {
  local name
  for name in ATOM_KV_OFFLOAD ATOM_KV_OFFLOAD_EXTRA_CONFIG; do
    if [[ -n "${!name:-}" ]]; then
      echo "[mooncake-store][FAIL] ${name} is set by the launcher under MOONCAKE_STORE=1; remove it from the case" >&2
      exit 2
    fi
  done
  local connector
  if connector="$(kv_transfer_offload_connector "${PREFILL_KV_TRANSFER_CONFIG:-}")"; then
    echo "[mooncake-store][FAIL] PREFILL_KV_TRANSFER_CONFIG names the offload connector ${connector}; under MOONCAKE_STORE=1 the launcher adds mooncake_store through ATOM_KV_OFFLOAD, and a worker runs one offload connector, so keep only the P/D connector there" >&2
    exit 2
  fi
  require_mooncake_qp_per_endpoint
  refuse_mooncake_nic_auto_discovery
  local layout="${ATOMESH_PD_WORKER_LAYOUT:-multi_node}" prefill_nodes="${xP:-1}" decode_nodes="${yD:-1}"
  if [[ -n "$(mooncake_setting MOONCAKE_STORE_DECODE_OWNERS)" ]] \
    && [[ "${layout}" != "multi_node" || "${prefill_nodes}" != "1" || "${decode_nodes}" != "1" ]]; then
    echo "[mooncake-store][FAIL] MOONCAKE_STORE_DECODE_OWNERS needs the decode on nodes of its own (pd_worker_layout multi_node, here ${layout}), one prefill node (here ${prefill_nodes}), whose masters they join, and one decode node (here ${decode_nodes}): the pools count the decode owners once" >&2
    exit 2
  fi
  if [[ ! "${MOONCAKE_STORE_LEASE_TTL_MS:-10000}" =~ ^[0-9]+$ ]] \
    || (( 10#${MOONCAKE_STORE_LEASE_TTL_MS:-10000} == 0 )); then
    echo "[mooncake-store][FAIL] MOONCAKE_STORE_LEASE_TTL_MS=${MOONCAKE_STORE_LEASE_TTL_MS:-} is not a positive number of milliseconds" >&2
    exit 2
  fi
  if [[ ! "${MOONCAKE_STORE_HOST_POOL_GIB:-0}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[mooncake-store][FAIL] MOONCAKE_STORE_HOST_POOL_GIB=${MOONCAKE_STORE_HOST_POOL_GIB:-} is not a number of GiB" >&2
    exit 2
  fi
  mooncake_store_worker_config >/dev/null
  plan_mooncake_store_owners
  plan_mooncake_store_pools
}

# The image must carry the Store: the master and owner binaries, which only a
# Mooncake built WITH_STORE=ON installs, and the mooncake.store module the
# prefill workers' connector imports. Without them a Store process would only
# "exit unexpectedly", or the workers fail late in their startup.
check_mooncake_store_image() {
  local binary
  for binary in mooncake_master mooncake_client; do
    if ! command -v "${binary}" >/dev/null 2>&1; then
      echo "[mooncake-store][FAIL] ${binary} is not on PATH: MOONCAKE_STORE=1 needs an image whose Mooncake is built WITH_STORE=ON" >&2
      exit 2
    fi
  done
  local import_error
  if ! import_error="$(python3 -c 'import mooncake.store' 2>&1)"; then
    printf '%s\n' "${import_error}" >&2
    echo "[mooncake-store][FAIL] python3 cannot import mooncake.store, which the prefill workers' mooncake_store connector needs" >&2
    exit 2
  fi
}

# Runs before this node starts anything: the prefill settings as start_prefill
# will see them, the decode's QP count as start_decode will, and the image.
# Each role env is applied in a subshell, so none of it stays in this shell.
validate_mooncake_store_settings() {
  # Before the switch check, so a decode-only MOONCAKE_STORE is caught too.
  local name
  while IFS='=' read -r name _; do
    if [[ "${name}" == ATOMESH_DECODE_ENV_MOONCAKE_STORE \
      || "${name}" == ATOMESH_DECODE_ENV_MOONCAKE_STORE_* ]]; then
      echo "[mooncake-store][FAIL] ${name}: the Mooncake Store settings belong to the prefill role env (ATOMESH_PREFILL_ENV_), which every node reads; in the decode role env only the decode node would see them" >&2
      exit 2
    fi
  done < <(env)
  mooncake_store_requested || return 0
  (
    apply_role_env "ATOMESH_PREFILL_ENV_" "${host_ip}"
    check_mooncake_store_settings
  )
  (
    # The decode node's owners start from the decode role env.
    apply_role_env "ATOMESH_DECODE_ENV_" "${host_ip}"
    require_mooncake_qp_per_endpoint
    refuse_mooncake_nic_auto_discovery
  )
  check_mooncake_store_image
}

# Every owner device needs an ACTIVE port, and one owner's devices must sit on
# one NUMA node: Mooncake spreads a segment over the nodes of its NICs,
# overriding numa_exec.py's binding.
check_mooncake_owner_devices() {
  local ib_root="${ATOMESH_IB_SYSFS_ROOT:-/sys/class/infiniband}"
  local i device node nodes
  for i in "${!mooncake_owner_plan_devices[@]}"; do
    nodes=""
    for device in ${mooncake_owner_plan_devices[i]//,/ }; do
      if [[ ! -e "${ib_root}/${device}" ]]; then
        echo "[mooncake-store][FAIL] owner${i} RDMA device ${device} is not in ${ib_root}" >&2
        exit 2
      fi
      if ! grep -qs '^4:' "${ib_root}/${device}"/ports/*/state; then
        echo "[mooncake-store][FAIL] owner${i} RDMA device ${device} has no ACTIVE port" >&2
        exit 2
      fi
      node="$(cat "${ib_root}/${device}/device/numa_node" 2>/dev/null || echo '?')"
      [[ " ${nodes} " == *" ${node} "* ]] || nodes+="${nodes:+ }${node}"
    done
    if [[ "${nodes}" == *" "* ]]; then
      echo "[mooncake-store][FAIL] owner${i}'s RDMA devices ${mooncake_owner_plan_devices[i]} sit on NUMA nodes ${nodes}: Mooncake would spread its segment over them instead of keeping it on node ${mooncake_owner_plan_numa[i]}; give each owner the devices of one node" >&2
      exit 2
    fi
  done
}

# prepare_mooncake_store_memory [prefill GPUs]: drops the clean page cache that
# the owners' huge-page faults would otherwise reclaim through compaction, then
# refuses pins that do not fit a NUMA node (numa_memory_budget.py warns when
# they exceed its free memory), and compacts each node for its pins: owners
# that fault a fragmented node from many threads get 4 KiB pages, which the
# NICs refuse. Each prefill GPU (default HIP_VISIBLE_DEVICES) pins
# MOONCAKE_STORE_HOST_POOL_GIB on its node; a decode node has none.
prepare_mooncake_store_memory() {
  local prefill_gpus="${1-${HIP_VISIBLE_DEVICES:-}}"
  local -a drop_dirs=()
  IFS=':' read -r -a drop_dirs <<< "$(mooncake_setting MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS "${MODEL_PATH}")"
  echo "[mooncake-store] dropping the page cache of ${drop_dirs[*]}"
  python3 "${ATOMESH_SCRIPT_DIR}/drop_page_cache.py" "${drop_dirs[@]}"
  local -a pins=()
  local i
  for i in "${!mooncake_owner_plan_numa[@]}"; do
    pins+=("${mooncake_owner_plan_numa[i]}:${mooncake_owner_plan_gib[i]}")
  done
  if ! python3 "${ATOMESH_SCRIPT_DIR}/numa_memory_budget.py" \
    --reserve-gib "$(mooncake_setting MOONCAKE_STORE_NODE_RESERVE_GIB 128)" \
    --gpus "${prefill_gpus}" \
    --per-gpu-gib "${MOONCAKE_STORE_HOST_POOL_GIB:-0}" \
    --compact \
    ${pins[@]+"${pins[@]}"}; then
    echo "[mooncake-store][FAIL] this node's Store owners and the prefill GPUs' host transfer pools (MOONCAKE_STORE_HOST_POOL_GIB per GPU) must fit their NUMA nodes, see numa-budget above" >&2
    exit 2
  fi
}

mooncake_store_running() {
  local pid
  for pid in ${mooncake_store_pids[@]+"${mooncake_store_pids[@]}"}; do
    [[ -n "${pid}" ]] && return 0
  done
  return 1
}

# Returns 1 and records the first exit status when a Store process exited.
reap_dead_mooncake_store() {
  local i rc any_died=0
  for i in "${!mooncake_store_pids[@]}"; do
    [[ -n "${mooncake_store_pids[i]}" ]] || continue
    process_is_running "${mooncake_store_pids[i]}" && continue
    set +e
    wait "${mooncake_store_pids[i]}"
    rc=$?
    set -e
    [[ "${rc}" -eq 0 ]] && rc=1
    mooncake_store_pids[i]=""
    mooncake_store_failed_rc="${mooncake_store_failed_rc:-${rc}}"
    tail -n 50 "${mooncake_store_logs[i]}" >&2 || true
    echo "[mooncake-store][FAIL] ${mooncake_store_names[i]} exited unexpectedly rc=${rc}, see ${mooncake_store_logs[i]}" >&2
    any_died=1
  done
  [[ "${any_died}" -eq 0 ]]
}

# Without the Store the prefill workers' lookups answer nothing and their saves
# fail, which they only log.
exit_if_mooncake_store_died() {
  reap_dead_mooncake_store || exit "${mooncake_store_failed_rc}"
}

# mooncake_metrics_file <pool>: where the pool master's final metrics go.
mooncake_metrics_file() {
  printf '%s' "${RUNTIME_LOG_DIR}/mooncake-master-rank-${NODE_RANK}$(mooncake_pool_suffix "$1").metrics"
}

# Saves each running master's metrics: the puts, gets, evictions and capacity
# of the whole run. cleanup_processes calls it before stopping anything: a
# decode node stops its owners once the router closes, and a master drops an
# owner client_ttl (10 s) after its last heartbeat.
save_mooncake_store_metrics() {
  [[ "${mooncake_store_metrics_saved}" == "0" ]] || return 0
  local pool
  for (( pool = 0; pool < mooncake_store_master_count; pool++ )); do
    [[ -n "${mooncake_store_pids[pool]:-}" ]] || continue
    curl -s --max-time 10 \
      "http://${host_ip}:$(mooncake_pool_port "${ATOMESH_MOONCAKE_METRICS_PORT}" "${pool}")/metrics" \
      > "$(mooncake_metrics_file "${pool}")" 2>/dev/null || true
  done
  mooncake_store_metrics_saved=1
}

# check_mooncake_store_capacity <pool>: records a failure when the pool
# master's final metrics count less than its owners' segments: an owner whose
# heartbeats lapse past client_ttl loses its segment while it keeps running,
# which nothing else reports.
check_mooncake_store_capacity() {
  local pool="$1" metrics counted want
  [[ "${mooncake_store_ready}" == "1" ]] || return 0
  metrics="$(mooncake_metrics_file "${pool}")"
  want="${mooncake_pool_capacity[pool]}"
  counted="$(awk '$1 == "master_total_capacity_bytes" { print $2 }' "${metrics}" 2>/dev/null || true)"
  if [[ -z "${counted}" ]]; then
    echo "[mooncake-store] WARNING: ${metrics} has no master_total_capacity_bytes; the owners' capacity at shutdown is unknown" >&2
    return 0
  fi
  if awk -v counted="${counted}" -v want="${want}" \
    'BEGIN { exit !(counted + 0 < want + 0) }'; then
    echo "[mooncake-store][FAIL] the master of pool ${pool} counts ${counted} of its owners' ${want} bytes at shutdown: an owner lost its segment during the run, see the owner logs" >&2
    mooncake_store_failed_rc="${mooncake_store_failed_rc:-1}"
  fi
}

# Returns non-zero when a Store process died before it was stopped, or a
# master lost owner capacity during the run.
stop_mooncake_store() {
  terminate_process_group "${mooncake_page_cache_dropper_pid}"
  mooncake_page_cache_dropper_pid=""
  mooncake_store_running || return 0
  save_mooncake_store_metrics
  reap_dead_mooncake_store || true
  local pool
  for (( pool = 0; pool < mooncake_store_master_count; pool++ )); do
    # A dead master answers for nobody.
    if [[ -n "${mooncake_store_pids[pool]:-}" ]]; then
      check_mooncake_store_capacity "${pool}"
    fi
  done
  local i
  # Owners before the master they report to.
  for (( i = ${#mooncake_store_pids[@]} - 1; i >= 0; i-- )); do
    [[ -n "${mooncake_store_pids[i]}" ]] || continue
    terminate_process_group "${mooncake_store_pids[i]}"
    mooncake_store_pids[i]=""
    echo "[mooncake-store] ${mooncake_store_names[i]} stopped, log ${mooncake_store_logs[i]}"
  done
  [[ -z "${mooncake_store_failed_rc}" ]]
}

# Fails startup (stopping what it started) unless <condition> holds within
# <timeout> seconds; a dying Store process fails it at once.
wait_for_mooncake_store() {
  local what="$1" timeout="$2"
  shift 2
  local deadline=$(( $(date +%s) + timeout ))
  local log
  until "$@"; do
    if ! reap_dead_mooncake_store; then
      echo "[wait][FAIL] mooncake-store ${what}: a Store process exited" >&2
      stop_mooncake_store || true
      exit "${mooncake_store_failed_rc}"
    fi
    for log in "${mooncake_store_logs[@]}"; do
      if grep -q -E "Failed to setup client|Failed to mount|Check failed|FATAL" "${log}" 2>/dev/null; then
        tail -n 50 "${log}" >&2 || true
        echo "[wait][FAIL] mooncake-store ${what}: see ${log}" >&2
        stop_mooncake_store || true
        exit 1
      fi
    done
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      for log in "${mooncake_store_logs[@]}"; do
        tail -n 20 "${log}" >&2 || true
      done
      echo "[wait][FAIL] mooncake-store ${what} not ready after ${timeout}s" >&2
      stop_mooncake_store || true
      exit 1
    fi
    sleep 2
  done
}

# mooncake_master_ready <addr> <pool>: the HTTP metadata server answers about
# a second before the master's RPC server listens, so the pool's master is
# ready once both do.
mooncake_master_ready() {
  local addr="$1" pool="$2"
  # Any HTTP answer, a 404 for the probe key included, means it is serving.
  curl -s -o /dev/null --max-time 5 \
    "http://${addr}:$(mooncake_pool_port "${ATOMESH_MOONCAKE_METADATA_PORT}" "${pool}")/metadata?key=atomesh_probe" \
    && timeout 5 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ \
      "${addr}" "$(mooncake_pool_port "${ATOMESH_MOONCAKE_MASTER_PORT}" "${pool}")" 2>/dev/null
}

# mooncake_masters_ready <addr>: every pool's master on <addr> is ready.
mooncake_masters_ready() {
  local pool
  for pool in "${!mooncake_pool_devices[@]}"; do
    mooncake_master_ready "$1" "${pool}" || return 1
  done
}

# mooncake_store_capacity_reaches <addr>: every pool's master on <addr> counts
# the segments of its owners on all nodes.
mooncake_store_capacity_reaches() {
  local pool
  for pool in "${!mooncake_pool_devices[@]}"; do
    curl -s --max-time 5 \
      "http://$1:$(mooncake_pool_port "${ATOMESH_MOONCAKE_METRICS_PORT}" "${pool}")/metrics" 2>/dev/null \
      | awk -v want="${mooncake_pool_capacity[pool]}" \
        '$1 == "master_total_capacity_bytes" { found = ($2 + 0 >= want) } END { exit !found }' \
      || return 1
  done
}

# mooncake_owner_pool <devices>: the pool of an owner on those devices.
mooncake_owner_pool() {
  local pool
  for pool in "${!mooncake_pool_devices[@]}"; do
    if [[ -z "${mooncake_pool_devices[pool]}" || "${mooncake_pool_devices[pool]}" == "$1" ]]; then
      echo "${pool}"
      return 0
    fi
  done
  echo "[mooncake-store][FAIL] no pool serves owner devices $1" >&2
  exit 2
}

# Drops the model's page cache every 45 s while the workers load weights, which
# would refill it; page cache is per file, so this serves every node.
start_mooncake_page_cache_dropper() {
  setsid python3 "${ATOMESH_SCRIPT_DIR}/drop_page_cache.py" \
    --every 45 --for "$(mooncake_setting MOONCAKE_STORE_PAGE_CACHE_DROP_SECONDS 1800)" "${MODEL_PATH}" &
  mooncake_page_cache_dropper_pid=$!
}

# start_mooncake_store_owners <master addr>: starts this node's owners
# (mooncake_owner_plan_*), each joining its pool's master on <master addr>.
start_mooncake_store_owners() {
  local master_addr="$1"
  # No GPU: a process that sees one makes Mooncake probe every buffer through HIP.
  # glibc's THP malloc makes the whole segment huge pages, which the NICs
  # register in under a second per 64 GiB MR; eno1 is firewalled, so the
  # transfer engine binds the role IP.
  local -a owner_env=(
    HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 MC_NUM_QP_PER_EP=1
    GLIBC_TUNABLES=glibc.malloc.hugetlb=1
    "MC_MAX_MR_SIZE=$(mooncake_setting MOONCAKE_STORE_OWNER_MAX_MR_SIZE 68719476736)"
    "MC_TCP_BIND_ADDRESS=${host_ip}"
  )
  local i pool log
  local -a cmd=()
  for i in "${!mooncake_owner_plan_numa[@]}"; do
    pool="$(mooncake_owner_pool "${mooncake_owner_plan_devices[i]}")"
    log="${RUNTIME_LOG_DIR}/mooncake-owner-rank-${NODE_RANK}-${i}.log"
    cmd=(
      python3 "${ATOMESH_SCRIPT_DIR}/numa_exec.py" "${mooncake_owner_plan_numa[i]}"
      mooncake_client
      --host="${host_ip}" --port="$(( ATOMESH_MOONCAKE_OWNER_PORT + i ))"
      --master_server_address="${master_addr}:$(mooncake_pool_port "${ATOMESH_MOONCAKE_MASTER_PORT}" "${pool}")"
      --metadata_server="http://${master_addr}:$(mooncake_pool_port "${ATOMESH_MOONCAKE_METADATA_PORT}" "${pool}")/metadata"
      --protocol=rdma --device_names="${mooncake_owner_plan_devices[i]}"
      --global_segment_size="$(( mooncake_owner_plan_gib[i] * 1024 * 1024 * 1024 ))"
      --local_buffer_size=0 --threads="$(mooncake_setting MOONCAKE_STORE_OWNER_THREADS 4)"
    )
    dump_launch_info "MOONCAKE_OWNER" "${owner_env[@]}" "${cmd[@]}"
    env "${owner_env[@]}" setsid "${cmd[@]}" >"${log}" 2>&1 &
    mooncake_store_pids+=("$!")
    mooncake_store_logs+=("${log}")
    mooncake_store_names+=("owner${i}")
  done
}

# Reports each owner of this node once it mounted: pid, node, NICs and how much
# of its memory is huge pages.
report_mooncake_store_owners() {
  local i pid
  for i in "${!mooncake_owner_plan_numa[@]}"; do
    pid="${mooncake_store_pids[mooncake_store_master_count + i]}"
    echo "[mooncake-store] owner${i} pid=${pid} numa=${mooncake_owner_plan_numa[i]} devices=${mooncake_owner_plan_devices[i]} $(grep -h AnonHugePages "/proc/${pid}/smaps_rollup" 2>/dev/null || echo 'AnonHugePages: ?')"
  done
}

# The connector's mooncake_store.pools for per_nic pools: each pool's NIC and
# its master.
mooncake_pools_json() {
  local pool sep=""
  printf '{'
  for pool in "${!mooncake_pool_devices[@]}"; do
    printf '%s"%s":{"master":"%s:%s","metadata":"http://%s:%s/metadata"}' "${sep}" \
      "${mooncake_pool_devices[pool]}" \
      "${host_ip}" "$(mooncake_pool_port "${ATOMESH_MOONCAKE_MASTER_PORT}" "${pool}")" \
      "${host_ip}" "$(mooncake_pool_port "${ATOMESH_MOONCAKE_METADATA_PORT}" "${pool}")"
    sep=","
  done
  printf '}'
}

# The connector keys the launcher derives from the Store it starts: the stage
# on a NIC reads that NIC's pool, whose owners share the NIC, under per_nic;
# otherwise every stage reads the one master and stays off the owners' NICs.
mooncake_store_launcher_config_json() {
  printf '{"mooncake_store.local_hostname":"%s","mooncake_store.protocol":"rdma",' "${host_ip}"
  if [[ -n "${mooncake_pool_devices[0]}" ]]; then
    printf '"mooncake_store.pools":%s}' "$(mooncake_pools_json)"
  else
    printf '"mooncake_store.master":"%s:%s",' "${host_ip}" "${ATOMESH_MOONCAKE_MASTER_PORT}"
    printf '"mooncake_store.metadata":"http://%s:%s/metadata",' "${host_ip}" "${ATOMESH_MOONCAKE_METADATA_PORT}"
    printf '"mooncake_store.owner_rdma_devices":"%s"}' "$(mooncake_owner_devices_csv)"
  fi
}

start_mooncake_store() {
  mooncake_store_requested || return 0
  # Several prefill workers in one shell share one Store, whose memory plan
  # and compaction covered the first worker's GPUs only.
  if mooncake_store_running; then
    if [[ "${mooncake_store_started_for}" != "${HIP_VISIBLE_DEVICES:-}" ]]; then
      echo "[mooncake-store][FAIL] the Store was planned for the prefill GPUs ${mooncake_store_started_for}; this prefill worker uses GPUs ${HIP_VISIBLE_DEVICES:-}" >&2
      stop_mooncake_store || true
      exit 2
    fi
    return 0
  fi
  check_mooncake_store_settings
  check_mooncake_owner_devices
  local worker_config
  worker_config="$(mooncake_store_worker_config "$(mooncake_store_launcher_config_json)")" || exit 2
  prepare_mooncake_store_memory
  mooncake_store_started_for="${HIP_VISIBLE_DEVICES:-}"
  start_mooncake_page_cache_dropper
  mooncake_store_pids=()
  mooncake_store_logs=()
  mooncake_store_names=()
  mooncake_store_ready=0
  mooncake_store_metrics_saved=0
  local -a store_env=(HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 MC_NUM_QP_PER_EP=1)
  local pool log
  local -a cmd=()
  for pool in "${!mooncake_pool_devices[@]}"; do
    log="${RUNTIME_LOG_DIR}/mooncake-master-rank-${NODE_RANK}$(mooncake_pool_suffix "${pool}").log"
    cmd=(
      python3 "${ATOMESH_SCRIPT_DIR}/numa_exec.py" "${MOONCAKE_STORE_MASTER_NUMA:-1}"
      mooncake_master
      --rpc_port="$(mooncake_pool_port "${ATOMESH_MOONCAKE_MASTER_PORT}" "${pool}")" --rpc_thread_num=32
      --enable_http_metadata_server=true
      --http_metadata_server_host="${host_ip}"
      --http_metadata_server_port="$(mooncake_pool_port "${ATOMESH_MOONCAKE_METADATA_PORT}" "${pool}")"
      --metrics_port="$(mooncake_pool_port "${ATOMESH_MOONCAKE_METRICS_PORT}" "${pool}")"
      --default_kv_lease_ttl="${MOONCAKE_STORE_LEASE_TTL_MS:-10000}"
      --eviction_high_watermark_ratio="${MOONCAKE_STORE_EVICTION_HIGH_WATERMARK:-0.90}"
      --eviction_ratio="${MOONCAKE_STORE_EVICTION_RATIO:-0.05}"
      --allocation_strategy=random --memory_allocator=offset --client_ttl=10
    )
    dump_launch_info "MOONCAKE_MASTER" "${store_env[@]}" "${cmd[@]}"
    env "${store_env[@]}" setsid "${cmd[@]}" >"${log}" 2>&1 &
    mooncake_store_pids+=("$!")
    mooncake_store_logs+=("${log}")
    mooncake_store_names+=("master$(mooncake_pool_suffix "${pool}")")
  done
  mooncake_store_master_count="${#mooncake_pool_devices[@]}"
  wait_for_mooncake_store masters "${MOONCAKE_STORE_MASTER_WAIT_TIMEOUT:-120}" \
    mooncake_masters_ready "${host_ip}"
  start_mooncake_store_owners "${host_ip}"
  local capacity=0
  for pool in "${!mooncake_pool_devices[@]}"; do
    capacity=$(( capacity + mooncake_pool_capacity[pool] ))
  done
  echo "[wait] mooncake-store pools=${#mooncake_pool_devices[@]} owners here=${#mooncake_owner_plan_numa[@]} capacity=${capacity} bytes"
  wait_for_mooncake_store "owners" "${MOONCAKE_STORE_WAIT_TIMEOUT:-1200}" \
    mooncake_store_capacity_reaches "${host_ip}"
  mooncake_store_ready=1
  report_mooncake_store_owners
  mooncake_store_prefill_env=(
    "ATOM_KV_OFFLOAD=mooncake_store"
    "ATOM_KV_OFFLOAD_EXTRA_CONFIG=${worker_config}"
    # One load worker by default, as PR #2459 settled on for this Store: four
    # hung its PP prefill pipeline at c96.
    "OFFLOAD_LOAD_WORKERS=${OFFLOAD_LOAD_WORKERS:-1}"
    "MC_NUM_QP_PER_EP=1"
    "MC_MAX_MR_SIZE=${MC_MAX_MR_SIZE:-1073741824}"
    "MC_TCP_BIND_ADDRESS=${host_ip}"
  )
  echo "[wait][OK] mooncake-store masters=${host_ip}:${ATOMESH_MOONCAKE_MASTER_PORT}+100/pool pools=${mooncake_pool_devices[*]:-shared} metrics=http://${host_ip}:${ATOMESH_MOONCAKE_METRICS_PORT}/metrics"
}

# On a decode node: starts MOONCAKE_STORE_DECODE_OWNERS on this node's memory,
# joining the prefill node's masters, before the decode workers load. Waits
# until every pool counts the owners of all nodes, as the prefill node does
# before its workers start.
start_mooncake_store_decode_owners() {
  mooncake_store_requested || return 0
  [[ -n "$(mooncake_setting MOONCAKE_STORE_DECODE_OWNERS)" ]] || return 0
  # The prefill node's own Store (a single-node job starts decode there too).
  mooncake_store_running && return 0
  plan_mooncake_store_pools
  plan_mooncake_store_owners MOONCAKE_STORE_DECODE_OWNERS
  check_mooncake_owner_devices
  prepare_mooncake_store_memory ""
  start_mooncake_page_cache_dropper
  mooncake_store_pids=()
  mooncake_store_logs=()
  mooncake_store_names=()
  mooncake_store_master_count=0
  local timeout
  timeout="$(mooncake_setting MOONCAKE_STORE_WAIT_TIMEOUT 1200)"
  wait_for_mooncake_store "prefill-node masters" "${timeout}" \
    mooncake_masters_ready "${NODE0_ADDR}"
  start_mooncake_store_owners "${NODE0_ADDR}"
  echo "[wait] mooncake-store decode owners=${#mooncake_owner_plan_numa[@]} joining ${NODE0_ADDR}"
  wait_for_mooncake_store "owners" "${timeout}" \
    mooncake_store_capacity_reaches "${NODE0_ADDR}"
  report_mooncake_store_owners
  echo "[wait][OK] mooncake-store decode owners mounted on ${NODE0_ADDR}'s pools ${mooncake_pool_devices[*]:-shared}"
}

cleanup_processes() {
  local rc=$?
  local pid
  # Before the router's end reaches a decode node, which then stops its owners.
  save_mooncake_store_metrics
  for pid in "$@"; do
    terminate_process_group "${pid}"
  done
  purge_lmcache_disk
  # Last: the workers read and write the Store until they exit.
  if ! stop_mooncake_store && [[ "${rc}" -eq 0 ]]; then
    rc="${mooncake_store_failed_rc}"
  fi
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
  local visible_devices="${6:-}"
  apply_role_env "ATOMESH_PREFILL_ENV_" "${host_ip}"
  if [[ -n "${visible_devices}" ]]; then
    export HIP_VISIBLE_DEVICES="${visible_devices}"
  fi
  reset_lmcache_disk
  start_mooncake_store
  # On the env command line, not exported: decode starts from this same shell.
  local -a prefill_offload_env=()
  if mooncake_store_running; then
    prefill_offload_env=("${mooncake_store_prefill_env[@]}")
  fi
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
  echo "[prefill] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} handshake=${handshake_port} dp_master=${dp_master_port} dp_base=${dp_base_port} cudagraph=${prefill_cudagraph_args[*]:-none}"
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
  dump_launch_info "PREFILL" "${prefill_offload_env[@]}" "${prefill_cmd[@]}"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" env "${prefill_cache_env[@]}" "${prefill_dp_env[@]}" "${prefill_offload_env[@]}" "${prefill_cmd[@]}"
}

start_decode() {
  local log_name="${1:-decode-rank-${NODE_RANK}}"
  local server_port="${2:-${DECODE_PORT}}"
  local handshake_port="${3:-${HANDSHAKE_PORT}}"
  local dp_master_port="${4:-${DECODE_DP_MASTER_PORT}}"
  local dp_base_port="${5:-${DECODE_DP_BASE_PORT}}"
  local visible_devices="${6:-}"
  apply_role_env "ATOMESH_DECODE_ENV_" "${host_ip}"
  if [[ -n "${visible_devices}" ]]; then
    export HIP_VISIBLE_DEVICES="${visible_devices}"
  fi
  start_mooncake_store_decode_owners
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
  # The P->D transfer engine on both sides reads MC_NUM_QP_PER_EP, which the
  # prefill's Mooncake Store sets to 1 (validate_mooncake_store_settings
  # refused any other value); endpoints with other QP counts cannot connect.
  local -a decode_mooncake_env=()
  if mooncake_store_requested; then
    decode_mooncake_env=("MC_NUM_QP_PER_EP=1")
  fi
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
  echo "[decode] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} handshake=${handshake_port} dp_master=${dp_master_port} dp_base=${dp_base_port} cudagraph=${decode_cudagraph_args[*]:-none}"
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
  dump_launch_info "DECODE" "${decode_mooncake_env[@]}" "${decode_cmd[@]}"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" env "${decode_cache_env[@]}" "${decode_dp_env[@]}" "${decode_mooncake_env[@]}" "${decode_cmd[@]}"
}

start_router() {
  echo "[router] prefill=${prefill_args[*]} decode=${decode_args[*]}"
  local mesh_binary="${ATOMESH_MESH_BINARY:-/app/ATOM/atom/mesh/target/release/atomesh}"
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
    # Respect explicit cache-aware routing for DPA workloads.
    if [[ "${router_policy}" != "cache_aware" ]]; then
      router_policy="dp_sticky"
    fi
    router_dp_aware_args=(--dp-aware)
  elif [[ "${#router_rank_mapping_args[@]}" -gt 0 ]]; then
    router_dp_aware_args=(--dp-aware)
  fi
  local -a router_policy_args=(--policy "${router_policy}")
  if [[ "${router_policy}" == "cache_aware" ]]; then
    # Keep the InferenceX defaults, with a case-level absolute-load override.
    router_policy_args+=(
      --prefill-policy cache_aware --decode-policy cache_aware
      --cache-threshold 0.8
      --balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD:-20}"
      --balance-rel-threshold 2.0
      --eviction-interval 300
    )
    router_rank_mapping_args=(--atom-pd-rank-mapping-policy "${ATOM_PD_RANK_MAPPING_POLICY}")
  fi
  local -a router_cmd=(
    "${mesh_binary}" launch
    --host 0.0.0.0
    --port "${ROUTER_PORT}"
    --pd-disaggregation
    "${prefill_args[@]}"
    "${decode_args[@]}"
    "${router_policy_args[@]}"
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
  fi
  # The compatibility entrypoint imports infx from the repository root.
  # Update cached checkouts too: older runs only populated utils/bench_serving.
  git -C "${bench_repo_dir}" sparse-checkout set utils/bench_serving infx
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
        --metric-percentiles='90,99' \
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
  python3 "${ATOMESH_SCRIPT_DIR}/../aiperf_dashboard.py" "$@"
}

write_aiperf_chrome_trace() {
  local out_dir="$1"
  local generator="${ATOMESH_SCRIPT_DIR}/../generate_aiperf_traces.py"
  local jsonl="${out_dir}/profile_export.jsonl"
  if [[ ! -f "${jsonl}" ]]; then
    echo "[aiperf] skip chrome trace: ${jsonl} was not produced"
    return 0
  fi
  if [[ ! -f "${generator}" ]]; then
    echo "[aiperf] skip chrome trace: ${generator} not found"
    return 0
  fi
  echo "[aiperf] converting ${jsonl} to Perfetto/Chrome trace"
  python3 "${generator}" "${out_dir}" \
    || echo "[aiperf] WARNING: chrome trace conversion failed for ${out_dir}" >&2
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
  local -a report_args=(
    --model "${MODEL_NAME} · ${DISPLAY_TOPOLOGY}"
    --mesh "127.0.0.1:${PROMETHEUS_PORT}"
  )
  local idx
  for idx in "${!prefill_ips[@]}"; do
    server_metrics_args+=("http://${prefill_ips[$idx]}:${prefill_ports[$idx]}/metrics")
    report_args+=(--prefill "${prefill_ips[$idx]}:${prefill_ports[$idx]}")
  done
  for idx in "${!decode_ips[@]}"; do
    server_metrics_args+=("http://${decode_ips[$idx]}:${decode_ports[$idx]}/metrics")
    report_args+=(--decode "${decode_ips[$idx]}:${decode_ports[$idx]}")
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
    local -a chat_template_args=()
    if (( AIPERF_BENCHMARK_DURATION < 900 )) \
      || [[ "${AIPERF_UNSAFE_OVERRIDE}" == "1" || "${AIPERF_UNSAFE_OVERRIDE}" == "true" ]]; then
      unsafe_args+=(--unsafe-override)
    fi
    if [[ "${AIPERF_APPLY_CHAT_TEMPLATE}" == "1" \
      || "${AIPERF_APPLY_CHAT_TEMPLATE}" == "true" ]]; then
      chat_template_args+=(--apply-chat-template)
    fi

    echo "[aiperf] ${result_file}"
    mkdir -p "${out_dir}"
    AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT="${AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT}" \
    AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT}" \
    AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES}" \
    AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT}" \
    AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT}" \
    AIPERF_UI_REALTIME_METRICS_ENABLED=true \
      python3 "${ATOMESH_SCRIPT_DIR}/observability/collect_metrics.py" \
      --output "${out_dir}/metrics" "${report_args[@]}" -- \
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
      "${chat_template_args[@]}" \
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
    write_aiperf_chrome_trace "${out_dir}"
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
      --log_samples \
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
    && ( "${EVAL_TASK}" == "swebench_lite" || "${EVAL_TASK}" == "gsm8k" ) \
    && ( "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1" ) ]]; then
    # Agentic performance cases require a fresh prefix/state-cache state. Run
    # their trace benchmark before any independent accuracy workload.
    run_benchmark
    run_eval
  else
    run_eval
    run_benchmark
  fi
}

validate_mooncake_store_settings
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
elif [[ "${PACKED_NODES_PD}" == "1" ]]; then
  worker_pids=()
  declare -A local_worker_pid=()
  # Save the base handshake port; each worker overrides it so
  # apply_role_env expands ${HANDSHAKE_PORT} to the per-worker value.
  PACKED_BASE_HANDSHAKE_PORT="${HANDSHAKE_PORT}"
  for idx in "${!prefill_nodes[@]}"; do
    [[ "${prefill_nodes[$idx]}" -eq "${NODE_RANK}" ]] || continue
    gpu_start="${prefill_gpus[$idx]%%,*}"
    export HANDSHAKE_PORT="$((PACKED_BASE_HANDSHAKE_PORT + gpu_start))"
    start_prefill "prefill-rank-${NODE_RANK}-worker-${idx}" "${prefill_ports[$idx]}" \
      "${HANDSHAKE_PORT}" \
      "$((PREFILL_DP_MASTER_PORT + idx * 400))" "$((PREFILL_DP_BASE_PORT + idx * 400))" \
      "${prefill_gpus[$idx]}"
    worker_pids+=("${server_pid}")
    local_worker_pid["prefill-${idx}"]="${server_pid}"
  done
  for idx in "${!decode_nodes[@]}"; do
    [[ "${decode_nodes[$idx]}" -eq "${NODE_RANK}" ]] || continue
    gpu_start="${decode_gpus[$idx]%%,*}"
    export HANDSHAKE_PORT="$((PACKED_BASE_HANDSHAKE_PORT + gpu_start))"
    start_decode "decode-rank-${NODE_RANK}-worker-${idx}" "${decode_ports[$idx]}" \
      "${HANDSHAKE_PORT}" \
      "$((DECODE_DP_MASTER_PORT + idx * 400))" "$((DECODE_DP_BASE_PORT + idx * 400))" \
      "${decode_gpus[$idx]}"
    worker_pids+=("${server_pid}")
    local_worker_pid["decode-${idx}"]="${server_pid}"
  done
  trap 'cleanup_processes ${router_pid:-} ${worker_pids[*]:-}' EXIT
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    for idx in "${!prefill_ips[@]}"; do
      wait_http "http://${prefill_ips[$idx]}:${prefill_ports[$idx]}/health" \
        "prefill-${prefill_ips[$idx]}:${prefill_ports[$idx]}" \
        "${WAIT_SERVER_TIMEOUT}" "${local_worker_pid["prefill-${idx}"]:-}"
    done
    for idx in "${!decode_ips[@]}"; do
      wait_http "http://${decode_ips[$idx]}:${decode_ports[$idx]}/health" \
        "decode-${decode_ips[$idx]}:${decode_ports[$idx]}" \
        "${WAIT_SERVER_TIMEOUT}" "${local_worker_pid["decode-${idx}"]:-}"
    done
    start_router
    wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "router" "${WAIT_ROUTER_TIMEOUT}"
    run_benchmark_and_eval
  else
    wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}"
    # Monitor all local workers while waiting for the router to shut down.
    for pid in "${worker_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        set +e; wait "${pid}"; rc=$?; set -e
        [[ "${rc}" -eq 0 ]] && rc=1
        echo "[wait][FAIL] packed worker ${pid} exited early rc=${rc}" >&2
        exit "${rc}"
      fi
    done
    wait_router_closed
  fi
  cleanup_processes "${router_pid:-}" "${worker_pids[@]}"
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
