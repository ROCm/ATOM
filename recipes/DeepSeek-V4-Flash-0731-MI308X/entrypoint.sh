#!/usr/bin/env bash
set -euo pipefail

export ATOM_DSV4_0731_OPTIMIZATIONS=${ATOM_DSV4_0731_OPTIMIZATIONS:-1}
MODEL_PATH=${MODEL_PATH:-/data/models/DeepSeek-V4-Flash-0731}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}
ONLINE_QUANT_CONFIG='{"layer_quant_config":{"*.experts":"per_block_fp8"}}'

exec python -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --server-port 8000 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --kv-cache-dtype bf16 \
  --index-cache-dtype fp8 \
  --method mtp \
  --num-speculative-tokens 6 \
  --cudagraph-mode FULL \
  --max-num-batched-tokens 131072 \
  --attn-prefill-chunk-size 16384 \
  --long-prefill-token-threshold 0 \
  --state-checkpoint-interval-tokens 8192 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.98 \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
  --no-enable-prefix-caching \
  "$@"
