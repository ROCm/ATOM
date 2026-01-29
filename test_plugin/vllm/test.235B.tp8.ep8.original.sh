#!/bin/bash
echo "run vllm original"

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_ROCM_USE_AITER_MHA=1

unset VLLM_ATTENTION_BACKEND
# export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0

# qucik allreduce for PoC
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

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
    2>&1 | tee log.serve.log &

    # --max-model-len 32768 \
    # --max-num-batched-tokens 16384 \
