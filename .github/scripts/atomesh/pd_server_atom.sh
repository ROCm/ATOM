#!/usr/bin/env bash
set -euo pipefail

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

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
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

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-2500}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

export ATOM_TORCH_PROFILER_DIR="${ATOM_TORCH_PROFILER_DIR:-${RUN_DIR}/online_quant/rank-${NODE_RANK}}"
mkdir -p "${RUN_DIR}"/{logs,benchmark_results,eval_results} "${ATOM_TORCH_PROFILER_DIR}"

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
    export "${name#${prefix}}=${value}"
  done < <(env)
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
  --no-enable_prefix_caching
)

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

write_metadata() {
  cat > "${RUN_DIR}/metadata-rank-${NODE_RANK}.json" <<EOF
{
  "rank": ${NODE_RANK},
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
  apply_prefixed_env "ATOMESH_PREFILL_ENV_" "${host_ip}"
  local -a prefill_cache_env=()
  build_server_cache_env "prefill" "${server_port}" prefill_cache_env
  echo "[prefill] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} handshake=${handshake_port} cudagraph=${PREFILL_CUDAGRAPH:-none}"
  local -a prefill_cmd=(
    python3 -m atom.entrypoints.openai_server
    "${server_common[@]}"
    --server-port "${server_port}"
    "${prefill_parallel[@]}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${handshake_port}}"
    "${prefill_cudagraph_args[@]}"
    ${PREFILL_SERVER_ARGS}
  )
  dump_launch_info "PREFILL" "${prefill_cmd[@]}"
  env "${prefill_cache_env[@]}" "${prefill_cmd[@]}" 2>&1 | tee "${RUN_DIR}/logs/${log_name}.log" &
  server_pid=$!
}

start_decode() {
  local log_name="${1:-decode-rank-${NODE_RANK}}"
  local server_port="${2:-${DECODE_PORT}}"
  apply_prefixed_env "ATOMESH_DECODE_ENV_" "${host_ip}"
  local max_conc
  max_conc="$(echo "${BENCH_MAX_CONCURRENCY}" | tr 'x,' '\n' | sort -n | tail -1)"
  local decode_max_num_seqs="${MAX_NUM_SEQS}"
  if [[ -n "${DECODE_MAX_NUM_SEQS}" ]]; then
    decode_max_num_seqs="${DECODE_MAX_NUM_SEQS}"
  fi
  if [[ "${ISL_LIST}" == "1024" && "${OSL}" == "1024" ]]; then
    decode_max_num_seqs="${max_conc}"
  fi
  local -a decode_cache_env=()
  build_server_cache_env "decode" "${server_port}" decode_cache_env
  echo "[decode] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} cudagraph=${DECODE_CUDAGRAPH:-none}"
  local -a decode_cmd=(
    python3 -m atom.entrypoints.openai_server
    "${server_common[@]}"
    --server-port "${server_port}"
    "${decode_parallel[@]}"
    --max-num-seqs "${decode_max_num_seqs}"
    --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}"
    "${decode_cudagraph_args[@]}"
    ${DECODE_SERVER_ARGS}
  )
  dump_launch_info "DECODE" "${decode_cmd[@]}"
  env "${decode_cache_env[@]}" "${decode_cmd[@]}" 2>&1 | tee "${RUN_DIR}/logs/${log_name}.log" &
  server_pid=$!
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

  local -a router_rank_mapping_args=()
  if [[ "${ATOM_PD_RANK_MAPPING_POLICY}" != "none" ]] \
    && has_cli_flag "${PREFILL_EXTRA_SERVER_ARGS}" "--enable-dp-attention" \
    && has_cli_flag "${DECODE_EXTRA_SERVER_ARGS}" "--enable-dp-attention"; then
    router_rank_mapping_args=(
      --atom-pd-rank-mapping-policy "${ATOM_PD_RANK_MAPPING_POLICY}"
      --dp-aware
    )
  fi
  local -a router_cmd=(
    /usr/local/bin/atomesh launch
    --host 0.0.0.0
    --port "${ROUTER_PORT}"
    --pd-disaggregation
    "${prefill_args[@]}"
    "${decode_args[@]}"
    --policy "${ROUTER_POLICY}"
    "${router_rank_mapping_args[@]}"
    --backend atom
    --log-level info
    --disable-circuit-breaker
    --prometheus-port "${PROMETHEUS_PORT}"
  )
  dump_launch_info "ROUTER" "${router_cmd[@]}"
  "${router_cmd[@]}" 2>&1 | tee "${RUN_DIR}/logs/router.log" &
  router_pid=$!
}

run_benchmark() {
  local bench_dir="/tmp/atomesh-bench-serving"
  if [[ ! -d "${bench_dir}/bench_serving" ]]; then
    rm -rf "${bench_dir}"
    mkdir -p "${bench_dir}"
    git clone --depth 1 https://github.com/kimbochen/bench_serving.git "${bench_dir}/bench_serving"
  fi
  IFS=',' read -r -a isls <<< "${ISL_LIST}"
  IFS=',' read -r -a concs <<< "${CONC_LIST}"
  local safe_model="${MODEL_NAME//\//-}"
  for isl in "${isls[@]}"; do
    for conc in "${concs[@]}"; do
      local result_file="pd-${BACKEND}-${safe_model}-${TOPOLOGY}-isl${isl}-osl${OSL}-conc${conc}-${RANDOM_RANGE_RATIO}.json"
      echo "[bench] ${result_file}"
      PYTHONDONTWRITEBYTECODE=1 python "${bench_dir}/bench_serving/benchmark_serving.py" \
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

run_eval() {
  [[ "${RUN_EVAL}" == "true" ]] || [[ "${RUN_EVAL}" == "1" ]] || return 0
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

write_metadata

if [[ "${NODE_RANK}" -eq 0 && "${SINGLE_NODE_PD}" == "1" ]]; then
  start_prefill "prefill-rank-0"
  prefill_pid="${server_pid}"
  start_decode
  decode_pid="${server_pid}"
  trap 'kill ${router_pid:-0} ${prefill_pid:-0} ${decode_pid:-0} 2>/dev/null || true' EXIT
  for ip in "${prefill_ips[@]}"; do
    wait_http "http://${ip}:${PREFILL_PORT}/health" "prefill-${ip}" "${WAIT_SERVER_TIMEOUT}" "${prefill_pid}"
  done
  for ip in "${decode_ips[@]}"; do
    wait_http "http://${ip}:${DECODE_PORT}/health" "decode-${ip}" "${WAIT_SERVER_TIMEOUT}" "${decode_pid}"
  done
  start_router
  wait_http "http://127.0.0.1:${ROUTER_PORT}/v1/models" "router" "${WAIT_ROUTER_TIMEOUT}"
  run_eval
  run_benchmark
  kill "${router_pid}" "${prefill_pid}" "${decode_pid}" 2>/dev/null || true
elif [[ "${NODE_RANK}" -eq 0 && "${PREFILL_SINGLE_NODE_PD}" == "1" ]]; then
  prefill_pids=()
  for idx in $(seq 0 $((xP - 1))); do
    gpu_start=$((idx * PREFILL_TP_SIZE))
    gpu_end=$((gpu_start + PREFILL_TP_SIZE - 1))
    export HIP_VISIBLE_DEVICES="$(seq -s, "${gpu_start}" "${gpu_end}")"
    prefill_port="${prefill_ports[$idx]}"
    handshake_port=$((HANDSHAKE_PORT + idx * PREFILL_TP_SIZE))
    start_prefill "prefill-rank-0-worker-${idx}" "${prefill_port}" "${handshake_port}"
    prefill_pids+=("${server_pid}")
  done
  trap 'kill ${router_pid:-0} ${prefill_pids[*]:-} 2>/dev/null || true' EXIT
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
  run_eval
  run_benchmark
  kill "${router_pid}" "${prefill_pids[@]}" 2>/dev/null || true
elif [[ "${NODE_RANK}" -eq 0 ]]; then
  start_prefill "prefill-rank-0"
  trap 'kill ${router_pid:-0} ${server_pid:-0} 2>/dev/null || true' EXIT
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
  run_eval
  run_benchmark
  kill "${router_pid}" "${server_pid}" 2>/dev/null || true
elif [[ "${DECODE_SINGLE_NODE_PD}" == "1" && "${NODE_RANK}" -eq "${xP}" ]]; then
  decode_pids=()
  for idx in $(seq 0 $((yD - 1))); do
    gpu_start=$((idx * DECODE_TP_SIZE))
    gpu_end=$((gpu_start + DECODE_TP_SIZE - 1))
    export HIP_VISIBLE_DEVICES="$(seq -s, "${gpu_start}" "${gpu_end}")"
    decode_port="${decode_ports[$idx]}"
    start_decode "decode-rank-${NODE_RANK}-worker-${idx}" "${decode_port}"
    decode_pids+=("${server_pid}")
  done
  trap 'kill ${decode_pids[*]:-} 2>/dev/null || true' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}"
  wait_router_closed
  kill "${decode_pids[@]}" 2>/dev/null || true
elif [[ "${PREFILL_SINGLE_NODE_PD}" == "1" ]]; then
  start_decode
  trap 'kill ${server_pid:-0} 2>/dev/null || true' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  kill "${server_pid}" 2>/dev/null || true
elif [[ "${NODE_RANK}" -lt "${xP}" ]]; then
  start_prefill "prefill-rank-${NODE_RANK}"
  trap 'kill ${server_pid:-0} 2>/dev/null || true' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  kill "${server_pid}" 2>/dev/null || true
else
  start_decode
  trap 'kill ${server_pid:-0} 2>/dev/null || true' EXIT
  wait_http "http://${NODE0_ADDR}:${ROUTER_PORT}/health" "router" "${WAIT_SERVER_TIMEOUT}" "${server_pid}"
  wait_router_closed
  kill "${server_pid}" 2>/dev/null || true
fi
