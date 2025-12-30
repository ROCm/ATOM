#!/bin/bash
set -euo pipefail

######################### CONFIGS ###########################
MODEL_PATH="/mnt/raid0/models/Meta-Llama-3-8B-Instruct"
######################### CONFIGS ###########################

if [ ! -e "$MODEL_PATH" ]; then
    echo "model '$MODEL_PATH' does not exist, please set MODEL_PATH firstly."
    exit 1
fi

python3 -m atom.entrypoints.openai_server \
        --model $MODEL_PATH \
        -tp 1 \
        --gpu-memory-utilization 0.9 \
        --server-port 8888