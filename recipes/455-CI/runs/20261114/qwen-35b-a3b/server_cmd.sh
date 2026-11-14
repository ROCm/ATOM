#!/bin/bash
export AITER_LOG_LEVEL=WARNING
HIP_VISIBLE_DEVICES=0 python3 -m atom.entrypoints.openai_server \
  --model /models/Qwen3.5-35B-A3B-MXFP4 \
  --tensor-parallel-size 1 --max-model-len 262144 --server-port 8000 \
  --method mtp --num-speculative-tokens 3 \
  --online_quant_config '{"layer_quant_config":{"mtp.layers.*.mlp.experts":"mxfp4"}}'
