#!/bin/bash
set -euo pipefail

######################### CONFIGS ###########################
MODEL_PATH="/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507_moe_w_mxfp4_a_mxfp4_kv_fp8"
######################### CONFIGS ###########################

#export AITER_LOG_MORE=2
#export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
#export HSA_ENABLE_DEBUG=1
#export HIP_LAUNCH_BLOCKING=1
#export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0

export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1

if [ ! -e "$MODEL_PATH" ]; then
    echo "model '$MODEL_PATH' does not exist, please set MODEL_PATH firstly."
    exit 1
fi

python3 -m atom.entrypoints.openai_server \
        --model $MODEL_PATH \
        --gpu-memory-utilization 0.9 \
        -tp 8 \
        --server-port 8888 \
        --kv_cache_dtype fp8 \
        --enable-expert-parallel \
        #--enforce-eager \
        #--level 0 \
        #--enable-expert-parallel 2>&1 | tee qwen.log
        #--torch-profiler-dir qwen_fp4_1k_1k
