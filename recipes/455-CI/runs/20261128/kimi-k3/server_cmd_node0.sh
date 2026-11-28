#!/bin/bash
RANK0=0 MASTER=<node0-ip>
# A/B 2026-11-22: k3_server_body_1118.sh for the ROCm 10.2.0 GA atom-dev image. Only change: the block that re-exported the
# 455-CI image's paths (PATH, ROCM_PATH=.../_rocm_sdk_core, LD_LIBRARY_PATH, LIBRARY_PATH, CPATH, HIP_* paths) is dropped, so the
# A/B image's own environment applies (ROCm at /opt/rocm, the gfx1250 atom venv). Everything else is verbatim.
set -euo pipefail
MODEL=/models/Kimi-K3
NIC=enp4s0np0
PORT=8000
[ "$(cat /proc/sys/kernel/numa_balancing)" = 0 ] || { echo "kernel.numa_balancing must be 0 on this tray" >&2; exit 1; }
# ---------------------------------------------------------------- environment of the image (re-exported explicitly)
export ROCM_HOME=$ROCM_PATH ROCM_SOURCE_DIR=$ROCM_PATH HIP_PATH=$ROCM_PATH
export USE_ROCM=1 MAX_JOBS=16 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export K3_HOME=/opt/k3 K3_ATOM_DIR=/app/ATOM
export ATOM_ENGRAM_MLOCK=1 ATOM_MEGA_MASK_PAD_ROWS=1 ATOM_MLA_QH128_ASM_PAGE_SIZE=64

# ---------------------------------------------------------------- architecture
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 GPU_TARGETS=gfx1250 MORI_GPU_ARCHS=gfx1250
export AITER_RUNTIME_GPU_ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0
export MORI_AMD_SMI_PATH=$ROCM_PATH/lib/libamd_smi.so.27

# ---------------------------------------------------------------- attention: MLA decode on the FlyDSL PS1 path
export ATOM_USE_TRITON_MLA=0 AITER_MLA_DECODE_PS1_FLYDSL=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ATOM_USE_UNIFIED_ATTN=1

# ---------------------------------------------------------------- MoE
export ATOM_MOE_GU_ITLV=1 ATOM_USE_TRITON_MOE_DECODE=0 AITER_USE_GROUPED_GEMM=1

# ---------------------------------------------------------------- GEMM, quantization, rope
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1

# ---------------------------------------------------------------- communication (RCCL only across the trays)
export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1 NCCL_GRAPH_REGISTER=0
export NCCL_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC MORI_SOCKET_IFNAME=$NIC
export AITER_CUSTOM_AR_USE_SYMM_MEM=0 ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DP_LM_HEAD_MODE=default

# ---------------------------------------------------------------- runtime / loading / caches
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1 HSA_XNACK=0        # HSA_XNACK=0 matches amdgpu noretry=1
export ATOM_LOADER_NUM_THREADS=4
export TRITON_CACHE_DIR=/root/.triton/cache HSA_COREDUMP_PATTERN=/root/.cache/gpucore.%p
export ATOM_PROFILER_TIMEOUT=3600 ROCPROFILER_REGISTER_FORCE_LOAD=0

# ---------------------------------------------------------------- server
ONLINE_QUANT='{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}'
cd /tmp   # started from /app, the image's aiter checkout directory would shadow the installed aiter package
echo "[k3 455-CI 1118] $(date '+%F %T') $(hostname): TP ranks $RANK0-$((RANK0 + 3)) of 8, master $MASTER"
exec python3 -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --served-model-name moonshotai/Kimi-K3 \
  --trust-remote-code \
  --server-port "$PORT" \
  -tp 8 \
  --tensor-parallel-size-local 4 \
  --tensor-parallel-rank "$RANK0" \
  --decode-context-parallel-size 8 \
  --data-parallel-master-ip "$MASTER" \
  --data-parallel-base-port 29700 \
  --kv_cache_dtype fp8 \
  --cudagraph-mode FULL \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --no-enable_prefix_caching \
  --online_quant_config "$ONLINE_QUANT" \
  --disable-uvicorn-access-log

