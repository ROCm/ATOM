#!/bin/bash
echo "run atom"
# export AMD_SERIALIZE_KERNEL=3

export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

rm -rf /root/.cache/

model_path=/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507-FP8

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.8 \
    --kv_cache_dtype fp8 \
    2>&1 | tee log.serve.log &
