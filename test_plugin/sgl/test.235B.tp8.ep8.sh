#!/bin/bash
echo "run sgl"

rm -rf /root/.cache/

export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

python3 -m sglang.launch_server \
    --model-path $model_path \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --model-impl atom \
    2>&1 | tee log.serve.log &
