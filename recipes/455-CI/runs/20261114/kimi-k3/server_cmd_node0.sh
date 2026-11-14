#!/bin/bash
RANK0=0 NODE0_IP=<NODE0_IP> NIC=enp4s0np0
# Recipe "## Kimi-K3 (2.8T)" server (455-CI 256124df). Needs RANK0, NODE0_IP, NIC. 1 TB trays: prefetch off.
export AITER_LOG_LEVEL=WARNING
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1 HSA_XNACK=1 ATOM_LOADER_NUM_THREADS=4
export ATOM_LOADER_PREFETCH=false    # 1 TB 内存的 tray
export ATOM_USE_TRITON_MLA=0 AITER_MLA_DECODE_PS1_FLYDSL=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ATOM_USE_UNIFIED_ATTN=1
export ATOM_USE_ASM_FP8_PREFILL_ATTN=1
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,1536,2048,3072,4096,5120,6144,7168,8192,10240,12288,14336,16384
export ATOM_MOE_GU_ITLV=1 ATOM_USE_TRITON_MOE_DECODE=0 AITER_USE_GROUPED_GEMM=1 AITER_FLYDSL_USE_HERD=1 ATOM_MONO_ENABLE=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 ATOM_FP4_MXSCALE_USE_32X4_SCALE_SHUFFLE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export MORI_SOCKET_IFNAME=$NIC NCCL_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC
export AITER_CUSTOM_AR_USE_SYMM_MEM=1 AITER_CUSTOM_AR_MAX_SIZE=8388608
export RCCL_DDA_LL_ONESHOT_THRESHOLD=2097152 RCCL_DDA_LL_TWOSHOT_THRESHOLD=134217728
export ATOM_USE_CUSTOM_ALL_GATHER=0
export NCCL_GRAPH_REGISTER=0

M=64
cd /tmp
python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --server-port 8000 \
  -tp 8 --tensor-parallel-size-local 4 --tensor-parallel-rank $RANK0 --decode-context-parallel-size 8 \
  --data-parallel-master-ip $NODE0_IP --data-parallel-base-port 29700 \
  --kv_cache_dtype fp8 --block-size 128 --cudagraph-mode FULL --gpu-memory-utilization 0.90 \
  --max-num-seqs $M --cudagraph-capture-sizes "[$(seq -s, 1 $M)]" \
  --max-num-batched-tokens 16384 --enable_prefix_caching --state-checkpoint-interval-tokens -1 \
  --method dspark --draft-model /models/Kimi-K3-DSpark --num-speculative-tokens 4 \
  --online_quant_config '{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}' \
  --disable_uvicorn_access_log
