# Shared settings for vLLM MiniMax-M3.1-Flash PD/agg AgentX runs. Source it, don't execute it.
# DRY_RUN=1 prints every command instead of starting it.

PD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=/shared/data/amd_int/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4
SERVED_MODEL_NAME=MiniMaxAI/MiniMax-M3.1-Flash
AGENTX_VENV=/app/agentx-venv
INFX_SRC=/app/InferenceX/inferencex-e2e
FRONT_PORT=8000 PREFILL_PORT=8100 DECODE_PORT=8200 BOOTSTRAP_PORT=8998
MOONCAKE_MASTER_PORT=19051 MOONCAKE_METRICS_PORT=19151
GiB=$((1024 ** 3))
LOG_DIR=${LOG_DIR:-$PWD/logs}

COMMON_ENV=(
  VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
  AITER_LOG_LEVEL=WARNING PYTHONPATH=/app/vllm-src PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
  VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
  HIP_FORCE_DEV_KERNARG=1 NCCL_DMABUF_ENABLE=0 HSA_ENABLE_IPC_MODE_LEGACY=1
  # Default 5s equals httpx's keepalive_expiry: the proxy reuses a socket the server is closing (ReadError).
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600
)
# HIP dmabuf MR registration costs an equal amount of extra VRAM per registered byte (OOM once KV > ~half of HBM).
MOONCAKE_ENV=(MC_GID_INDEX=1 MC_IB_TC=104 MC_MAX_CONCURRENT_REG_MR=1 VLLM_MOONCAKE_CONNECTOR_TIMEOUT=600 MOONCAKE_DISABLE_HIP_DMABUF=1)

# vllm_args CONC: set VLLM_ARGS to the `vllm serve` argv every server shares.
# cpunode N: argv prefix pinning to NUMA node N's CPUs (numactl --cpunodebind=N; numactl isn't installed).
cpunode() { echo taskset -c "$(< "/sys/devices/system/node/node$1/cpulist")"; }

vllm_args() {
  VLLM_ARGS=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0
    --tensor-parallel-size 4 --trust-remote-code --language-model-only
    --moe-backend aiter --block-size 128 --disable-hybrid-kv-cache-manager
    --kv-cache-dtype mxfp4 --attention-config '{"indexer_kv_dtype":"mxfp4"}'
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":4.04}'
    --max-model-len 1048576 --gpu-memory-utilization 0.9 --enable-prefix-caching
    --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice
    --max-num-seqs $((2 * $1))
  )
}

_PLAIN='^[A-Za-z0-9_./:=,@%+-]+$'
# quote ARGS...: shell-quote for humans (JSON stays readable in single quotes).
quote() {
  local a out=()
  for a; do [[ $a =~ $_PLAIN ]] && out+=("$a") || out+=("'$a'"); done
  echo "${out[*]}"
}

# start_bg NAME LOG CMD...: run CMD as its own process group (pgid = pid), logging to LOG,
# and append "pid name log" to $LOG_DIR/pids for wait/cleanup.
start_bg() {
  local name=$1 log=$2
  shift 2
  echo "[$name] $(quote "$@")"
  [[ ${DRY_RUN:-0} == 1 ]] && return 0
  echo "+ $(quote "$@")" > "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  echo "$! $name $log" >> "$LOG_DIR/pids"
}

# stop_all: TERM, then KILL, every process group in $LOG_DIR/pids; then check the GPUs are free.
stop_all() {
  local f=$LOG_DIR/pids pid alive _
  [[ -s $f ]] || return 0
  while read -r pid _; do kill -TERM -- "-$pid" 2>/dev/null; done < "$f"
  for _ in $(seq 60); do
    alive=0
    while read -r pid _; do pgrep -g "$pid" > /dev/null && alive=1; done < "$f"
    ((alive)) || break
    sleep 1
  done
  while read -r pid _; do kill -KILL -- "-$pid" 2>/dev/null; done < "$f"
  mv "$f" "$f.stopped"
  for _ in $(seq 60); do
    # rocm-smi shows host PIDs, so match on VRAM instead; tiny stray KFD clients (<1GiB) are ignored.
    rocm-smi --showpids 2> /dev/null | awk '$1 ~ /^[0-9]+$/ && $4 > 2^30 {busy=1} END {exit busy}' \
      && { echo "GPUs free"; return 0; }
    sleep 1
  done
  echo "WARNING: KFD processes still hold GPUs:" >&2
  rocm-smi --showpids >&2
  return 1
}
