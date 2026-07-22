#!/usr/bin/env bash
# Kimi-K3 (Minimax-m3-xiaobing) serving on MI355 (gfx950) x8, tp=8, CUDA-graph decode.
#
# Validated: gsm8k 0.94 flexible/strict; decode ~20 tok/s (~4x over eager).
#
# CUDA-graph note: capture is enabled simply by NOT passing --enforce-eager while
# keeping --level 0. ATOM treats model_type=kimi_linear as GDN (use_gdn=True) and,
# with cudagraph_mode=FULL, captures the whole eager decode forward per decode batch
# size (<= max-num-seqs). No torch.compile / @support_torch_compile is needed.
# gpu-memory-utilization is 0.93 (not 0.90) so the ~1.5GB cudagraph pool fits
# alongside the KDA per-request state cache; at 0.90 startup fails with
# "Per-request cache tensor exceeds available KV budget".
#
# Correctness flags that MUST stay:
#   -tp 8                              (tp4 OOMs: MoE weights ~175GB/GPU)
#   --no-enable_prefix_caching        SDPA MLA prefill only sees in-batch tokens;
#   --no-enable_chunked_prefill       a cached/chunked prefix would be missed.
#   (MLA layers use the flash KV-cache layout fix — commit 5240fd5 — to dodge the
#    aiter SHUFFLE-read bug at head_dim=192.)
#
# Env overrides: K3_GMU (default 0.93), K3_LEVEL (default 0), K3_PORT (default 8000),
#   K3_ENFORCE_EAGER=1 to fall back to pure eager (disables CUDA graphs).
set -euo pipefail

MODEL="${K3_MODEL:-/shared/data/amd_int/models/xiaobing/Minimax-m3-xiaobing}"
ATOM_DIR="${K3_ATOM_DIR:-/workdir/xiaobing/ATOM-K3}"

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# gfx950/MI355 kernel selection
export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=1
export ATOM_USE_UNIFIED_ATTN=1
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_SYNC_AFTER_LOAD=1
export ATOM_DIST_TIMEOUT_SECONDS=3600
# Small Kimi prefill batches fault on the route-indexed preshuffle grouped-gemm;
# route them through the regular fused grouped path.
export AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD="${AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD:-512}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-16}"

# Fast load: parallel page-cache prewarm (host has plenty RAM) + threadpool loader.
# (~3 min vs ~30 min; the default fastsafetensors path is serial and slower here.)
export ATOM_USE_FASTSAFETENSORS=0
export ATOM_LOADER_USE_THREADPOOL=1
export ATOM_LOADER_THREADPOOL_WORKERS="${ATOM_LOADER_THREADPOOL_WORKERS:-8}"
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"

EAGER_FLAG=()
if [[ "${K3_ENFORCE_EAGER:-0}" == "1" ]]; then
  EAGER_FLAG+=(--enforce-eager)   # disables CUDA graphs (debug only)
fi

# Prewarm the safetensors shards into the OS page cache in parallel.
( cd "$MODEL" 2>/dev/null && ls model-*.safetensors 2>/dev/null \
    | xargs -P12 -I{} dd if={} of=/dev/null bs=4M 2>/dev/null ) || true

cd "$ATOM_DIR"
exec /opt/venv/bin/python -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --kv_cache_dtype bf16 -tp 8 \
  --trust-remote-code \
  --max-model-len "${K3_MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${K3_MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${K3_MAX_NUM_BATCHED_TOKENS:-7168}" \
  --gpu-memory-utilization "${K3_GMU:-0.93}" \
  --server-port "${K3_PORT:-8000}" \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}' \
  --no-enable_prefix_caching \
  "${EAGER_FLAG[@]}"
