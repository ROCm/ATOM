#!/bin/bash
echo "run atom + vllm"
export VLLM_ATTENTION_BACKEND=CUSTOM

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# for debug
# export AMD_SERIALIZE_KERNEL=3
# export AMD_LOG_LEVEL=2

export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

# export ROCM_DEBUG_AGENT_OPTIONS="--all"
# export HSA_TOOLS_LIB=/opt/rocm-7.0.0/lib/librocm-debug-agent.so.2
# export TRITON_DEBUG=1

rm -rf /root/.cache/

model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 18432 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --model-impl atom \
    2>&1 | tee log.serve.log &

    # --kv-cache-dtype fp8 \
    # --max-model-len 32768 \
    # --max-num-batched-tokens 16384 \
    # --no-enable-chunked-prefill \
    # --enable-expert-parallel \
