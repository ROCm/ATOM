#!/bin/bash
echo "run atom original"
# export AMD_SERIALIZE_KERNEL=3

export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

rm -rf /root/.cache/

model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --kv_cache_dtype fp8 \
    --max-num-batched-tokens 18432 \
    --max-model-len 16384 \
    --cudagraph-capture-sizes "[1,2,4,8,16,32,48,64,128,256,512]" \
    2>&1 | tee log.serve.log &

    # --max-model-len 32768 \
    # --max-num-batched-tokens 16384 \
