#!/usr/bin/env bash
set -euo pipefail

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export ATOM_USE_TRITON_GEMM="${ATOM_USE_TRITON_GEMM:-1}"
export AITER_USE_GROUPED_GEMM="${AITER_USE_GROUPED_GEMM:-1}"
export ATOM_USE_UNIFIED_ATTN="${ATOM_USE_UNIFIED_ATTN:-1}"
export ATOM_FORCE_ATTN_TRITON="${ATOM_FORCE_ATTN_TRITON:-1}"
export ATOM_SYNC_AFTER_LOAD="${ATOM_SYNC_AFTER_LOAD:-1}"
export ATOM_DIST_TIMEOUT_SECONDS="${ATOM_DIST_TIMEOUT_SECONDS:-3600}"
export ATOM_LOADER_USE_THREADPOOL="${ATOM_LOADER_USE_THREADPOOL:-0}"
export ATOM_LOADER_THREADPOOL_WORKERS="${ATOM_LOADER_THREADPOOL_WORKERS:-1}"
export ATOM_USE_FASTSAFETENSORS="${ATOM_USE_FASTSAFETENSORS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-16}"
# The gfx1250 route-indexed preshuffle kernel faults on small Kimi prefill
# batches (for example 96 tokens). Keep large warmup/batches on contiguous-M,
# but route small batches through the regular fused grouped path.
export AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD="${AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD:-512}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
EXTRA_ARGS=()
if [[ "${LOAD_DUMMY:-0}" == "1" || "${LOAD_DUMMY:-0}" == "true" || "${LOAD_DUMMY:-0}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--load_dummy)
fi

exec python -m atom.entrypoints.openai_server \
  --model /mnt/models/Kimi-K3 \
  --kv_cache_dtype bf16 -tp 4 \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --level 0 \
  "${EXTRA_ARGS[@]}"