#!/bin/bash

export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1

python -m atom.entrypoints.openai_server \
  --model /mnt/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --enable-dp-attention \
  --gpu-memory-utilization 0.85
