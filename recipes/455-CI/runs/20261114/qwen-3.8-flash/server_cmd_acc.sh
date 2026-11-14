#!/bin/bash
export AITER_LOG_LEVEL=WARNING
HIP_VISIBLE_DEVICES=0 ENABLE_CK=0 GPU_ARCHS=gfx1250 \
ATOM_MOE_GU_ITLV=1 ATOM_GDN_STATE_MIDSTEP=1 \
python3 -m atom.entrypoints.openai_server \
  --model /models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --served-model-name /models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --host 0.0.0.0 --server-port 8000 --trust-remote-code \
  -tp 1 \
  --max-num-seqs 160 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.95 \
  --enable_prefix_caching \
  --timeout-keep-alive 60 \
  --method mtp \
  --num-speculative-tokens 3 \
  --online_quant_config '{"layer_quant_config":{"mtp.layers.0.mlp.experts*":"mxfp4"}}' \
  --kv_cache_dtype fp8 \
  --index-cache-dtype bf16 \
  --block-size 16 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128]'
