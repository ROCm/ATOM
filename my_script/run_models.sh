#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# gfx1250 CK kernels: native aiter CK modules (quant/cache/rmsnorm/moe_asm/sample/
# custom_all_reduce) now build+work on gfx1250 via the CK patch
# aiter-k3/patches/ck_gfx1250_enable.patch (registers gfx1250 as a gfx12-family
# CK target). Apply it at aiter build: `git -C 3rdparty/composable_kernel apply
# patches/ck_gfx1250_enable.patch`. The torch/opus/triton fallbacks below default
# OFF (native); set them to 1 to fall back if the patch is not applied.
export ATOM_USE_TRITON_GEMM="${ATOM_USE_TRITON_GEMM:-1}"
export AITER_USE_GROUPED_GEMM="${AITER_USE_GROUPED_GEMM:-1}"
export ATOM_USE_UNIFIED_ATTN="${ATOM_USE_UNIFIED_ATTN:-1}"
export ATOM_FORCE_ATTN_TRITON="${ATOM_FORCE_ATTN_TRITON:-1}"
export ATOM_SYNC_AFTER_LOAD="${ATOM_SYNC_AFTER_LOAD:-1}"
export ATOM_DIST_TIMEOUT_SECONDS="${ATOM_DIST_TIMEOUT_SECONDS:-3600}"
export ATOM_LOADER_USE_THREADPOOL="${ATOM_LOADER_USE_THREADPOOL:-0}"
export ATOM_LOADER_THREADPOOL_WORKERS="${ATOM_LOADER_THREADPOOL_WORKERS:-1}"
export ATOM_USE_FASTSAFETENSORS="${ATOM_USE_FASTSAFETENSORS:-1}"
export ATOM_FASTSAFETENSORS_DIST_LOAD="${ATOM_FASTSAFETENSORS_DIST_LOAD:-0}"
export ATOM_FASTSAFETENSORS_DEVICE="${ATOM_FASTSAFETENSORS_DEVICE:-cuda}"
export ATOM_FASTSAFETENSORS_NOGDS="${ATOM_FASTSAFETENSORS_NOGDS:-1}"
export AITER_DISABLE_CUSTOM_ALL_REDUCE="${AITER_DISABLE_CUSTOM_ALL_REDUCE:-0}"
export ATOM_USE_CUSTOM_ALL_GATHER="${ATOM_USE_CUSTOM_ALL_GATHER:-0}"
export AITER_USE_TRITON_QUANT="${AITER_USE_TRITON_QUANT:-0}"
export AITER_USE_OPUS_RMSNORM="${AITER_USE_OPUS_RMSNORM:-0}"
export AITER_USE_TORCH_RMSNORM="${AITER_USE_TORCH_RMSNORM:-0}"
export AITER_USE_TORCH_TOPK="${AITER_USE_TORCH_TOPK:-0}"
export AITER_DISABLE_GROUPED_A8W4="${AITER_DISABLE_GROUPED_A8W4:-0}"
export ATOM_USE_TORCH_SAMPLER="${ATOM_USE_TORCH_SAMPLER:-0}"
export ATOM_USE_TORCH_CACHE="${ATOM_USE_TORCH_CACHE:-0}"
export ATOM_KDA_FORCE_RECURRENT="${ATOM_KDA_FORCE_RECURRENT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-16}"
# NOTE: the AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD workaround is no longer set
# here — aiter grouped_moe_gfx1250 now defaults the contiguous-M scheduler OFF on
# gfx1250 (it has an OOB bug at large spread routing), using the correct
# non-contiguous path. Set the env explicitly only to override that default.

# gfx1250 Kimi-K3 accuracy fixes:
#  - whole-prompt prefill (--no-enable_chunked_prefill/--no-enable_prefix_caching
#    below) so the SDPA MLA sees the full sequence.
#  - ATOM_K3_MOE_CHUNK: REQUIRED for correctness. The gfx1250 grouped MoE + MXFP4
#    projections are only numerically correct at small M (large M: contiguous-M
#    OOB-crashes, non-contiguous returns wrong values -> gsm8k 0.0). MoE is
#    per-token independent, so sub-batch it to <=N tokens. Remove once the gfx1250
#    MoE kernel is fixed at large M.
#  - ATOM_WARMUP_MAX_TOKENS: benign warmup-only cap (dummy warmup samples garbage
#    over all positions and faults the sampler; real inference samples last-token
#    only). Remove once dummy warmup skips sampling.
# The aiter grouped_moe_gfx1250 contiguous-M-off default is kept as a defensive
# guard (with MoE-chunk, M stays <=chunk so it rarely matters).
export ATOM_K3_MOE_CHUNK="${ATOM_K3_MOE_CHUNK:-128}"
export ATOM_WARMUP_MAX_TOKENS="${ATOM_WARMUP_MAX_TOKENS:-256}"

# batched >= longest prompt (gsm8k 5-shot ~633) so a prompt is prefilled whole
# (no chunking -> SDPA MLA sees the full sequence). 4096 model len fits gsm8k.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
# 0.93 leaves room for the CUDA-graph decode pool alongside the KDA state cache.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"

# CUDA-graph decode is ON by default (~4x decode vs eager): keep --level 0 and do
# NOT pass --enforce-eager. Set K3_ENFORCE_EAGER=1 to fall back to pure eager.
EAGER_FLAG=()
if [[ "${K3_ENFORCE_EAGER:-0}" == "1" ]]; then
  EAGER_FLAG+=(--enforce-eager)
fi

EXTRA_ARGS=()
if [[ "${LOAD_DUMMY:-0}" == "1" || "${LOAD_DUMMY:-0}" == "true" || "${LOAD_DUMMY:-0}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--load_dummy)
fi
EXTRA_ARGS+=("$@")

exec python -m atom.entrypoints.openai_server \
  --model /data/models/Kimi-K3 \
  --kv_cache_dtype bf16 -tp 4 \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --level 0 \
  --no-enable_prefix_caching \
  --no-enable_chunked_prefill \
  "${EAGER_FLAG[@]}" \
  "${EXTRA_ARGS[@]}"