# export AITER_QUICK_REDUCE_QUANTIZATION=INT4
# export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
# export ATOM_USE_GLUON_PA_DECODE=1
# export HIP_VISIBLE_DEVICES=6,7

# vllm serve /workspace/shared/data/amd_int/models/MiniMax-M2.5 \
#     --host localhost \
#     --port 8100 \
#     --async-scheduling \
#     --load-format fastsafetensors \
#     --tensor-parallel-size 2 \
#     --trust-remote-code \
#     --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
#     --kv-cache-dtype fp8 \
#     --max-num-batched-tokens 16384 \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.9 \
#     --no-enable-prefix-caching \
#     --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": "False"}' \
    # --enforce-eager

# model_path=/workspace/shared/data/amd_int/models/MiniMax-M3-MXFP4/

# model_path=/workspace/shared/data/amd_int/models/MiniMax-M3
export HIP_VISIBLE_DEVICES=0,1,2,3
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_M3_DUMP_DIR="./minimax_dump/"
# export HSA_ENABLE_SDMA=1
# export HSA_USE_SVM=1
# export HSA_XNACK=1
# export ATOM_USE_TRITON_MOE=1
# export ATOM_USE_TRITON_GEMM=1
# export ENABLE_CK=0 
# export AITER_USE_OPUS_MOE_SORTING=1
# export ATOM_USE_UNIFIED_ATTN=0


# python -m atom.entrypoints.openai_server \
#   --model $model_path \
#   -tp 4 --server-port 8013 --trust-remote-code --gpu-memory-utilization 0.7  \
#   --block-size 128 \
#   --no-enable_prefix_caching \
  # --torch-profiler-dir ./trace --mark-trace
main_model=/workspace/shared/data/amd_int/models/MiniMax-M3-MXFP4
# draft_model=/workspace/shared/data/amd_int/models/MiniMax-M3-EAGLE3
# export HIP_VISIBLE_DEVICES=0,1,2,3
python -m atom.entrypoints.openai_server --model $main_model \
  -tp 4 --server-port 8014 --trust-remote-code --gpu-memory-utilization 0.8 --block-size 128 --no-enable_prefix_caching \
  --max-num-batched-tokens 32768 --max-model-len 32768 --max-num-seqs 128 --enforce-eager --level 0
  # --kv_cache_dtype fp8 \
  # --torch-profiler-dir ./trace
  # --method eagle3 --draft-model $draft_model --num-speculative-tokens 3 \

  # --enforce-eager


#!/usr/bin/env bash
# set -euo pipefail

# rm -f "${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"/modelinfos/*minimax_m3* 2>/dev/null || true
# ok
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LOG_FILE="$SCRIPT_DIR/minimax-m3-preview-2.log"

# unset VL MM_BATCHED MM_ENCODER_ATTN

# export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# # export HIP_VISIBLE_DEVICES="4,5,6,7"
# export ATOM_USE_TRITON_MOE="${ATOM_USE_TRITON_MOE:-1}"

# MODEL="/shared/data/amd_int/models/MiniMax-M3"
# SERVED_NAME="${SERVED_NAME:-MiniMax-M3}"
# PORT="${PORT:-8000}"
# TP="${TP:-8}"
# GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# # GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
# MAX_LEN="${MAX_LEN:-16384}"
# MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
# # MAX_SEQS="${MAX_SEQS:-8}"
# ATTN="TRITON_ATTN"
# LOAD_FORMAT="${LOAD_FORMAT:-auto}"
# SKIP_TOKENIZER_INIT="${SKIP_TOKENIZER_INIT:-0}"
# ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# EXTRA_ARGS=()
# if [[ "$SKIP_TOKENIZER_INIT" == "1" ]]; then
#     EXTRA_ARGS+=(--skip-tokenizer-init)
# fi
# if [[ "$ENFORCE_EAGER" == "1" ]]; then
#     EXTRA_ARGS+=(--enforce-eager)
# fi

# echo "### serve: model=$MODEL"
# echo "### serve: devices=$HIP_VISIBLE_DEVICES tp=$TP port=$PORT max_len=$MAX_LEN"
# echo "### serve: log=$LOG_FILE"
# export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
# export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
# vllm serve "$MODEL" \
#     --dtype bfloat16 \
#     --load-format "$LOAD_FORMAT" \
#     --host localhost \
#     --port "$PORT" \
#     --tensor-parallel-size "$TP" \
#     --gpu-memory-utilization "$GPU_MEM_UTIL" \
#     --max-model-len "$MAX_LEN" \
#     --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
#     --block-size 128 \
#     --no-enable-prefix-caching \
#     --language-model-only \
#     --no-trust-remote-code \
#     --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
#     "${EXTRA_ARGS[@]}" \
#     2>&1 | tee "$LOG_FILE"

# export HIP_VISIBLE_DEVICES=4,5,6,7
# AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
# HSA_ENABLE_SDMA=1 \
# HSA_USE_SVM=1 \
# HSA_XNACK=1 \
# AITER_DISABLE_KERNARG_PRELOAD=1 \
# ATOM_USE_TRITON_MOE=1 \
# ATOM_FORCE_ATTN_TRITON=1 \
# ATOM_LOADER_USE_THREADPOOL=0 \
# AITER_ROPE_TRITON_BACKEND=1 \
# ATOM_ENABLE_DS_QKNORM_FUSION=0 \
# ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0 \
# ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0 \
# ATOM_USE_TRITON_GEMM=1 \
# ENABLE_DS_QKNORM_FUSION=0 \
# ENABLE_CK=0 \
# AITER_USE_OPUS_MOE_SORTING=1 \
# ATOM_USE_UNIFIED_ATTN=0 \
# python -m atom.entrypoints.openai_server \
#   --model /workspace/shared/data/amd_int/models/MiniMax-M3-MXFP4 \
#   --server-port 8013 \
#   --trust-remote-code \
#   -tp 4 \
#   --gpu-memory-utilization 0.8 \
#   --block-size 128 \
#   --max-model-len 32768 \
#   --max-num-seqs 128 \
#   --max-num-batched-tokens 32768 \
#   --torch-profiler-dir /app/trace \
#   --no-enable_prefix_caching