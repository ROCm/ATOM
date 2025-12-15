#!/bin/bash

export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
export HSA_ENABLE_DEBUG=1

export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1


#MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
MODEL=/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507_moe_w_mxfp4_a_mxfp4_kv_fp8
rm -rf /root/.cache/atom/
python -m atom.entrypoints.openai_server --model ${MODEL} -tp 8 --kv_cache_dtype fp8 --enable-expert-parallel

