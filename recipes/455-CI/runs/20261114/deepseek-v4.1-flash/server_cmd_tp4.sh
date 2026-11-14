#!/bin/bash
export AITER_LOG_LEVEL=WARNING
cd /app/ATOM
export HIP_VISIBLE_DEVICES=0,1,2,3
export ATOM_DSPARK_PROB_DRAFT=1 ATOM_DSPARK_BLOCK_VERIFY=1
export AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608 ATOM_GC_THRESHOLD=20000,50,50 HSA_XNACK=1
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1
export ATOM_MONO_ENABLE=1
python3 -u -m atom.entrypoints.openai_server --model /models/DeepSeek-V4.1-Flash --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 -tp 4 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --block-size 16 \
  --level 0 --method dspark --num-speculative-tokens 5 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]" \
  --disable-uvicorn-access-log
