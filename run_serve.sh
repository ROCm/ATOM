#!/usr/bin/env bash
set -euo pipefail

rm -f "${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"/modelinfos/*minimax_m3* 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/minimax-m3-preview.log"

unset VL MM_BATCHED MM_ENCODER_ATTN

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export ATOM_USE_TRITON_MOE="${ATOM_USE_TRITON_MOE:-1}"

MODEL="/shared/data/amd_int/models/MiniMax-M3"
SERVED_NAME="${SERVED_NAME:-MiniMax-M3}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-16384}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
# MAX_SEQS="${MAX_SEQS:-8}"
ATTN="TRITON_ATTN"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
SKIP_TOKENIZER_INIT="${SKIP_TOKENIZER_INIT:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EXTRA_ARGS=()
if [[ "$SKIP_TOKENIZER_INIT" == "1" ]]; then
    EXTRA_ARGS+=(--skip-tokenizer-init)
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
fi

echo "### serve: model=$MODEL"
echo "### serve: devices=$HIP_VISIBLE_DEVICES tp=$TP port=$PORT max_len=$MAX_LEN"
echo "### serve: log=$LOG_FILE"
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1



    # --attention-backend "$ATTN" \
#  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/mnt/raid0/xiaobing/m3/scripts/minimax-m3-preview_profiler_trace", "torch_profiler_record_shapes": true, "torch_profiler_with_stack": true}' \
vllm serve "$MODEL" \
    --dtype bfloat16 \
    --load-format "$LOAD_FORMAT" \
    --host localhost \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --block-size 128 \
    --kv-cache-dtype bf16 \
    --no-enable-prefix-caching \
    --language-model-only \
    --no-trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
 