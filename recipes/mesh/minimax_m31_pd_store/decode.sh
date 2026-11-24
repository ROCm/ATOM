#!/usr/bin/env bash
# Decode (Mooncake consumer) on GPUs 4-7 (NUMA1). Run inside the decode container.
#   MODE=dep4  -tp 4 --enable-dp-attention --enable-expert-parallel: 4 DP ranks, each
#              TP1 attention over its own requests, MoE EP4 over MoRI (default)
#   MODE=mega  dep4 + --moe-backend mega (aiter MegaMoEV2), MNBT 512 so its combine wire stays bf16
#   MODE=tp4   -tp 4 with indexer CP (ATOM_M3_INDEXER_CP=1): today's decode
#   FORCE_AL=4.04 benchmark runs, empty for real acceptance
#   D_SEQS per DP rank under dep4/mega (default 64), D_MEM (0.9)
#   D_PENDING_KV_FRACTION (0.15), D_RESERVED_DECODE_TOKENS (512): decode admission
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODE=${MODE:-dep4}
rm -rf /root/.cache/atom/*
export HIP_VISIBLE_DEVICES=4,5,6,7
export ATOM_DP_MASTER_PORT=29700 ATOM_DP_BASE_PORT=29800
# Decode admission (recipes/pd_disaggregation_guide.md): pending-receive KV budget per D rank
# and the decode growth kept free for every admitted request (>= 2 x (k + 1) = 16 for k=7).
D_PENDING_KV_FRACTION=${D_PENDING_KV_FRACTION:-0.15}
D_RESERVED_DECODE_TOKENS=${D_RESERVED_DECODE_TOKENS:-512}
KV="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_consumer\",\"handshake_port\":6301,\"protocol\":\"rdma\",\"ib_device\":\"rdma4,rdma5,rdma6,rdma7\",\"pd_admission\":{\"enabled\":true,\"max_pending_kv_fraction\":${D_PENDING_KV_FRACTION},\"num_reserved_decode_tokens\":${D_RESERVED_DECODE_TOKENS}}}"
# Per DP rank under DP attention: requests, CUDA-graph batch sizes and MNBT.
# Under DP + a drafter the decode token graphs stop at ATOM_TOKEN_GRAPH_DP_MAX_TOKENS
# (512 rows = 64 requests x 8 verify rows), so 64 requests per rank is the graph ceiling.
CG64="[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]"
case "${MODE}" in
  dep4)
    PAR="-tp 4 --enable-dp-attention --enable-expert-parallel --dp-load-balance least_requests"
    unset ATOM_M3_INDEXER_CP
    SEQS=${D_SEQS:-64}; MNBT=4096; CG=${D_CG:-${CG64}}
    export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-8G} ;;
  mega)
    PAR="-tp 4 --enable-dp-attention --enable-expert-parallel --dp-load-balance least_requests --moe-backend mega"
    unset ATOM_M3_INDEXER_CP
    SEQS=${D_SEQS:-64}; MNBT=512; CG=${D_CG:-${CG64}}
    export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-8G} ;;
  tp4)
    PAR="-tp 4"
    SEQS=${D_SEQS:-256}; MNBT=16384
    export ATOM_M3_INDEXER_CP=1
    CG=${D_CG:-[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,144,160,176,192,208,224,240,256]} ;;
  *) echo "MODE must be dep4, mega or tp4" >&2; exit 2 ;;
esac
env | grep -E '^(ATOM_|AITER_|MC_|MORI_|HIP_|NCCL_|RCCL_|GLOO_)' | sort > "${RUN}/decode-${MODE}.env"
# shellcheck disable=SC2086
exec python3 -m atom.entrypoints.openai_server \
  --model "${MODEL}" --trust-remote-code --host 0.0.0.0 \
  --server-port "${D_PORT}" --port 18106 \
  ${PAR} \
  --kv_cache_dtype "${KV_DTYPE}" --index-cache-dtype "${KV_DTYPE}" --block-size 128 \
  --max-model-len 1048576 --max-num-seqs "${SEQS}" --max-num-batched-tokens "${MNBT}" \
  --gpu-memory-utilization "${D_MEM:-0.9}" \
  --cudagraph-capture-sizes "${CG}" \
  ${SPEC_ARGS} ${AL_ARGS} \
  --kv-transfer-config "${KV}" \
  --timeout-keep-alive 900 ${D_EXTRA:-} > "${RUN}/decode-${MODE}.log" 2>&1
