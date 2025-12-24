#!/bin/bash
set -euo pipefail

######################### CONFIGS ###########################
MODEL_PATH="/mnt/raid0/models/deepseek-ai/DeepSeek-V3.2-Exp/"
######################### CONFIGS ###########################

export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0

if [ ! -e "$MODEL_PATH" ]; then
    echo "model '$MODEL_PATH' does not exist, please set MODEL_PATH firstly."
    exit 1
fi

python3 -m atom.entrypoints.openai_server \
        --model /mnt/raid0/models/deepseek-ai/DeepSeek-V3.2-Exp/ \
        --gpu-memory-utilization 0.8 \
        --max-num-batched-tokens 70000 \
        -tp 8 \
        --server-port 8888 \
        --block-size 16
