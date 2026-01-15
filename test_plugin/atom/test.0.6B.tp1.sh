#!/bin/bash
echo "run atom original"
export CUDA_VISIBLE_DEVICES=7
# export AMD_SERIALIZE_KERNEL=3

rm -rf /root/.cache/

model_path=/data/pretrained-models/Qwen3-0.6B

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.1 \
    --kv_cache_dtype fp8 \
    --level 0 \
    --enforce-eager \
    2>&1 | tee log.serve.log &
