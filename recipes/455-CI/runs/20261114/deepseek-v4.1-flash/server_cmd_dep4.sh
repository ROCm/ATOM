#!/bin/bash
export AITER_LOG_LEVEL=WARNING
cd /app/ATOM
R=/app/endpoints-recipe-dsv41
export HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 ATOM_DISABLE_MMAP=true AITER_LOG_LEVEL=WARNING
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1 HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1 HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27 MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250
export HIP_FORCE_DEV_KERNARG=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1
export ATOM_GC_THRESHOLD=20000,50,50 AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608
export MORI_SHMEM_HEAP_SIZE=17179869184
export ATOM_DSPARK_PROB_DRAFT=1 ATOM_DSPARK_BLOCK_VERIFY=1
export ATOM_DP_SESSION_AFFINITY=1
export ATOM_ENGRAM_UVA=1 ATOM_ENGRAM_UVA_COPY=0
export ATOM_PREFILL_DECODE_INTERVAL=24
export ATOM_MORI_V2=1 MEGA_DISPATCH_WIRE=fp4
export ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_STEP_TIMER=1 ATOM_STEP_TIMER_EVERY=500
export MAX_JOBS=16
export ATOM_DSV41_ROPE_HEAD_BLOCK=1
export ATOM_DECODER_GRAPH_SIZES=1,2,3,4,8,16
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072,4096,5120,6144
export AITER_CONFIG_GEMM_BF16="/app/aiter/aiter/configs/bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsr1_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsv41_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/flux2_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm47_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm53_flash_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/inkling_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimi_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/minimax_m3_eagle_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/minimax_m3_mxfp4_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/muse_glimmer_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/q3vl_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_5_397b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_8_27b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_8b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/wan22_bf16_tuned_gemm.csv:$R/configs/dsv41_bf16_all_tuned.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE="/app/aiter/aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:$R/configs/dsv41_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.minus_recipe9.csv:/app/aiter/aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/minimax_m3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimik3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:$R/configs/dsv41_dep4_dense_mx32_tuned.csv"
export AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE="/app/aiter/aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv:/app/aiter/aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv:$R/configs/dsv41_dep4_bmm_mx32_tuned.csv"
export AITER_MEGA_S1F=1 AITER_S1F_BLOCK_RUNS=4,2,8,8 AITER_S1F_MAX_RECV=4096 AITER_S1F_ANY_TILE=1
export AITER_TDM_COMPACT_ROUTE_UPTO=1024 AITER_TDM_COMPACT_ROUTE_SPEC=384,16
export ATOM_PREFILL_ADMIT=adaptive ATOM_PREFILL_ADMIT_VERSION=2
export ATOM_ENGRAM_MLOCK=1 ATOM_HYI_TPL_EXECUTOR=1 ATOM_HYI_DROP_FINISHED_SEQS=1 ATOM_HYI_ENC_MEMO=1
export ATOM_DSPARK_PROB_FUSED_CHAIN=1
export ATOM_DP_LOGITS_PEER_EXCHANGE=1
export AITER_MOE_LOC=1
python3 -u -m atom.entrypoints.openai_server --model /models/DeepSeek-V4.1-Flash --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 -tp 4 \
  --kv_cache_dtype fp4 --index-cache-dtype fp4 --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --block-size 16 \
  --level 0 --max-num-seqs 64 --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]" \
  --enable-expert-parallel --enable-dp-attention --all2all-backend high-throughput \
  --method dspark --num-speculative-tokens 5 \
  --disable-uvicorn-access-log --api-server-count 4
