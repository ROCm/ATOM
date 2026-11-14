#!/bin/bash
# Recipe "## DeepSeek R1" server with one change: --no-enable-chunked-prefill removed (agentic prompts exceed 16384 tokens).
export AITER_LOG_LEVEL=WARNING
env ATOM_MLA_ABSORB_BF16_FUSE_QOUT=1 ATOM_MLA_KV_SHUFFLED=1 ATOM_MEGA_MASK_PAD_ROWS=1 MEGA_COMBINE_WIRE=bf16 \
  AITER_DISABLE_KERNARG_PRELOAD=1 HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 ATOM_DP_LM_HEAD_MODE=allgather \
  ATOM_USE_UNIFIED_ATTN=1 ATOM_USE_TRITON_MLA=0 ATOM_MLA_PAGE_SIZE=1 AITER_MLA_DECODE_PS1_FLYDSL=1 \
  AITER_ROPE_TRITON_BACKEND=1 ATOM_ENABLE_DS_QKNORM_FUSION=0 ATOM_LOADER_NUM_THREADS=4 \
  ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0 ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0 ATOM_USE_TRITON_GEMM=1 \
  ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ENABLE_CK=0 AITER_USE_GROUPED_GEMM=1 AITER_FORCE_A8W4=0 \
  AITER_USE_OPUS_MOE_SORTING=1 ATOM_MOE_GU_ITLV=1 ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1 ATOM_MEGA_STAGE1_FUSED=0 \
  MEGA_DISPATCH=mori MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 AITER_FLYDSL_GATHER_REDUCE_WAVE_MIN_TOKENS=-1 \
  ATOM_GC_THRESHOLD=20000,50,50 MEGA_DISPATCH_WIRE=fp4 \
python3 -m atom.entrypoints.openai_server -tp 4 --model /models/Deepseek-S3_sq_a05_v2_mlperf6_1/ \
  --kv_cache_dtype fp8 --block-size 64 --no-enable_prefix_caching \
  --enable-expert-parallel --enable-dp-attention \
  --online_quant_config '{"layer_quant_config":{"*.self_attn.q_b_proj":"mxfp4"}}' \
  --max-num-seqs 2048 --gpu-memory-utilization 0.975 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,48,64,128,256,512,1024,1536,2048]'
