#!/bin/bash
export AITER_LOG_LEVEL=WARNING
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 AITER_QUICK_REDUCE_QUANTIZATION=INT4 ATOM_FORCE_ATTN_TRITON=1
export ATOM_GC_THRESHOLD=20000,50,50 ATOM_PA_FLYDSL=1
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
export ATOM_MOE_GU_ITLV=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1 AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export ATOM_MONO_ENABLE=0                  # gfx1250 上开 mono 时 decode 一步要几百 ms
export ATOM_M3_INDEXER_CP=1                # 并发 15 及以上用
# ATOM main 的 loader 通过全局符号表调用 omp_set_num_threads，不预加载 libgomp 时加载权重会让每个 rank 退出
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
cd /tmp
python3 -u -m atom.entrypoints.openai_server \
  --model /models/MiniMax-M3-MXFP4 --served-model-name /models/MiniMax-M3-MXFP4 \
  --host 0.0.0.0 --port 8896 --server-port 8890 --tensor-parallel-size 4 --trust-remote-code \
  --kv_cache_dtype fp8 --gpu-memory-utilization 0.95 --block-size 128 \
  --max-num-batched-tokens 32768 --attn-prefill-chunk-size 16384 --max-num-seqs 128 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --enable-prefix-caching \
  --method eagle3 --draft-model /models/MiniMax-M3-EAGLE3-GQA --num-speculative-tokens 3 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]'
