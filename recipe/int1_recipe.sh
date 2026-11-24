# int1 integrated recipe (source me). Server knobs for serve.sh / gate.sh / ladder.sh / decode_bench.sh.
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Mono for <= 16-row steps is wanted (user 05:34); default stays 0 until int1+mono passes the gate + mono_blocked lands
export ATOM_MONO_ENABLE=${MONO:-0}
export PYPATH_PRE=${PYPATH_PRE:-}  # amd_kernels is vendored at $REPO/amd_kernels, already on PYTHONPATH via $REPO itself; set PYPATH_PRE to your aiter checkout path before sourcing this chain
# sched
export ATOM_M3_PREFILL_TOKEN_GRAPH=1 ATOM_SCHED_PREFILL_DUTY=${DUTY:-0.3} ATOM_SCHED_DECODE_PREFILL_CAP=8192
# attn-prefill (bitwise)
export ATOM_M3_PREFILL_FAST=1 ATOM_M3_PREFILL_ATTN=ak16 ATOM_M3_PREFILL_INSERT=amdk
# amd-kernels survey: decode insert, decode index+attn (<=16 reqs), MXFP8 GEMMs
export ATOM_M3_AK_INSERT=1 ATOM_M3_AK_DECODE=1 ATOM_AK_MXFP8_GEMM=1
# amd-kernels MoE block (MXFP4 activations at all M)
export ATOM_M3_AK_MOE=${AK_MOE:-1}
# fused top-16 + NVFP4 sparse decode (> 16 reqs)
export ATOM_M3_FUSED_DECODE=1
# gemm tuned tables
export AITER_CONFIG_GEMM_BF16="$(dirname "${BASH_SOURCE[0]}")/configs/bf16_tuned_gemm_m31.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE="$(dirname "${BASH_SOURCE[0]}")/configs/mx8_tuned_gemm_m31.csv"
CG='--cudagraph-capture-sizes [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]'
DS='--dspark-config {"draft_kv_dtype":"fp8"}'
export EXTRA="$CG $DS ${EXTRA_MORE:-}"
