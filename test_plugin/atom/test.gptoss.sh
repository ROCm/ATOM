#!/bin/bash
echo "run atom"
# export CUDA_VISIBLE_DEVICES=7
# export AMD_SERIALIZE_KERNEL=3

export ATOM_GPT_OSS_MODEL=1

rm -rf /root/.cache/

model_path=/home/zejchen/gptoss

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --tensor-parallel-size 8 \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.8 \
    2>&1 | tee log.serve.log &
