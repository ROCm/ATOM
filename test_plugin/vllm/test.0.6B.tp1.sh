#!/bin/bash
echo "run vllm"
export CUDA_VISIBLE_DEVICES=7
# export AMD_SERIALIZE_KERNEL=3

export VLLM_ATTENTION_BACKEND=CUSTOM

# disable all the optimization for now
export ATOM_ENABLE_QK_NORM_ROPE_FUSION=0
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/data/pretrained-models/Qwen3-0.6B

vllm serve $model_path \
    --host localhost \
    --port 9090 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.1 \
    --async-scheduling \
    --load-format fastsafetensors \
    --no-enable-chunked-prefill \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --model-impl atom \
    --kv-cache-dtype fp8 \
    2>&1 | tee log.serve.log &
