#!/bin/bash
set -euo pipefail

######################### CONFIGS ###########################
MODEL_PATH="/mnt/raid0/models/gpt-oss-120b"
######################### CONFIGS ###########################

if [ ! -e "$MODEL_PATH" ]; then
    echo "model '$MODEL_PATH' does not exist, please set MODEL_PATH firstly."
    exit 1
fi

python3 -m atom.entrypoints.openai_server \
        --model $MODEL_PATH \
        -tp 1 \
        --kv_cache_dtype fp8 \
        --gpu-memory-utilization 0.9 \
        --server-port 8888