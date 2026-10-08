#!/bin/bash
set -euo pipefail

: "${DPRANK:?set DPRANK to this node first global DP rank}"

MASTER_IP="${MASTER_IP:-10.210.11.14}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-16}"
DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
FAKE_EPLB="${FAKE_EPLB:-1}"
MORI_FUSED="${MORI_FUSED:-1}"
SESSION_AFFINITY="${SESSION_AFFINITY:-1}"
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-0}"
PREFILL_DELAYER_TARGET_FILL="${PREFILL_DELAYER_TARGET_FILL:-0.9}"
BF16_GEMM_CONFIG="${BF16_GEMM_CONFIG:-}"
ONLINE_QUANT_CONFIG="${ONLINE_QUANT_CONFIG:-}"
USE_FLYDSL_GATHER="${USE_FLYDSL_GATHER:-1}"
USE_TRITON_GEMM="${USE_TRITON_GEMM:-0}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-0}"
SPEC_DECODE_ACCEPTANCE_LENGTH="${SPEC_DECODE_ACCEPTANCE_LENGTH:-}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-0}"
STATE_CHECKPOINT_INTERVAL_TOKENS="${STATE_CHECKPOINT_INTERVAL_TOKENS:-8192}"
STATE_CHECKPOINT_DEMAND="${STATE_CHECKPOINT_DEMAND:-1}"
ENABLE_REPLAYSSM="${ENABLE_REPLAYSSM:-0}"
MEGA_COMBINE_WIRE="${MEGA_COMBINE_WIRE:-bf16}"
DP_LB_REQ_EQUIV="${DP_LB_REQ_EQUIV:-512}"
ENABLE_PREFILL_DELAYER="${ENABLE_PREFILL_DELAYER:-1}"

export PYTORCH_ROCM_ARCH=gfx1250 AITER_RUNTIME_GPU_ARCH=gfx1250
export GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0
export ENABLE_CK=0

export ATOM_USE_TRITON_MLA=1
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ="${USE_FLYDSL_GATHER}"
export ATOM_UNFUSED_GATHER_KV_B_PROJ=0
export ATOM_USE_AITER_TRITON_ATTN=1 ATOM_USE_UNIFIED_ATTN=1

export ATOM_MOE_GU_ITLV=1
export ATOM_USE_TRITON_MOE_DECODE=0
export MEGA_DISPATCH=mori MEGA_DISPATCH_WIRE=fp4
# Existing stock infrastructure, held constant across A/B. This is not a
# recipe-owned optimization and does not require experimental_mori_aiter.patch.
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED="${MORI_FUSED}"
export ATOM_MEGA_COMBINE_WIRE="${MEGA_COMBINE_WIRE}"
export AITER_USE_GROUPED_GEMM=1 AITER_USE_OPUS_MOE_SORTING=1

export ATOM_USE_TRITON_GEMM="${USE_TRITON_GEMM}" ATOM_WO_A_USE_FLYDSL=1
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1

export NCCL_MNNVL_ENABLE=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS
export NCCL_CUMEM_ENABLE=1
export ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_USE_CUSTOM_ALL_GATHER=1 AITER_CUSTOM_AR_USE_SYMM_MEM=1
export MORI_SOCKET_IFNAME=enp1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1
export GLOO_SOCKET_IFNAME=enp1s0f1

export ATOM_DP_SESSION_AFFINITY="${SESSION_AFFINITY}"
export ATOM_DP_LB_REQ_EQUIV="${DP_LB_REQ_EQUIV}"
export ATOM_ENABLE_PREFILL_DELAYER="${ENABLE_PREFILL_DELAYER}"
export ATOM_PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL}"
export ATOM_PREFILL_DELAYER_TARGET_FILL="${PREFILL_DELAYER_TARGET_FILL}"
export ATOM_STATE_CHECKPOINT_DEMAND="${STATE_CHECKPOINT_DEMAND}"
export ATOM_ENABLE_REPLAYSSM="${ENABLE_REPLAYSSM}"
export HSA_XNACK=1 HSA_USE_SVM=1 HSA_ENABLE_SDMA=1
export ATOM_LOADER_USE_THREADPOOL=1 ATOM_LOADER_NUM_THREADS=4
if [[ -n "${BF16_GEMM_CONFIG}" ]]; then
  export AITER_CONFIG_GEMM_BF16="${BF16_GEMM_CONFIG}"
fi

if [[ "${USE_FLYDSL_GATHER}" == "1" ]]; then
  python3 -c '
from atom.utils import envs
from aiter.ops.flydsl import gather_kv_b_proj_flydsl
assert envs.ATOM_USE_FLYDSL_GATHER_KV_B_PROJ
assert callable(gather_kv_b_proj_flydsl)
'
fi

if [[ "${MEGA_COMBINE_WIRE}" != "bf16" ]]; then
  python3 -c '
import inspect
from atom.model_ops.fused_moe.mori_v2_prepare_finalize import _import_mega

mega = _import_mega()
assert "combine_quant" in inspect.signature(mega.__init__).parameters, (
    "ATOM_MEGA_COMBINE_WIRE=fp8/fp4 requires ROCm/AITER #5176 "
    "(commit 22ab77eb19d3); the installed AITER predates quantized combine"
)
assert "combine_quant" in inspect.signature(mega.__call__).parameters, (
    "installed MegaMoEGfx1250 cannot select a quantized combine wire per step"
)
'
fi

args=(
  --model /models/Kimi-K3
  --served-model-name moonshotai/Kimi-K3
  --trust-remote-code
  -tp 1
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
  --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}"
  --data-parallel-rank "${DPRANK}"
  --data-parallel-master-ip "${MASTER_IP}"
  --data-parallel-master-port 29500
  --data-parallel-base-port 29700
  --enable-expert-parallel
  --enable-dp-attention
  --kv_cache_dtype fp8
  --index-cache-dtype fp8
  --cudagraph-mode FULL
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}"
  --state-checkpoint-interval-tokens "${STATE_CHECKPOINT_INTERVAL_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --disable_uvicorn_access_log
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  args+=(--enable_prefix_caching)
else
  args+=(--no-enable_prefix_caching)
fi

if [[ "${FAKE_EPLB}" == "1" ]]; then
  args+=(--fake-eplb)
fi

if [[ -n "${ONLINE_QUANT_CONFIG}" ]]; then
  args+=(--online_quant_config "${ONLINE_QUANT_CONFIG}")
fi

if [[ "${NUM_SPECULATIVE_TOKENS}" != "0" ]]; then
  : "${DRAFT_MODEL_PATH:?set DRAFT_MODEL_PATH when speculative decoding is enabled}"
  args+=(
    --method dspark
    --draft-model "${DRAFT_MODEL_PATH}"
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
  )
  if [[ -n "${SPEC_DECODE_ACCEPTANCE_LENGTH}" ]]; then
    args+=(--spec-decode-acceptance-length "${SPEC_DECODE_ACCEPTANCE_LENGTH}")
  fi
fi

if [[ -n "${CUDAGRAPH_CAPTURE_SIZES}" ]]; then
  args+=(--cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}")
fi

if [[ -n "${TORCH_PROFILER_DIR:-}" ]]; then
  args+=(--torch-profiler-dir "${TORCH_PROFILER_DIR}")
fi

cd /tmp
exec python3 -m atom.entrypoints.openai_server "${args[@]}"
